import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        return_debug_info: bool = False,
    ) -> _model.Actions | tuple[_model.Actions, dict]:
        observation = _model.preprocess_observation(None, observation, train=False)
        
        # Always split RNG to maintain consistent behavior
        subtask_rng, action_rng = jax.random.split(rng, 2)
        
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(action_rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        
        # Return actions only - debug info handling moved outside JIT
        return x_0

    def sample_actions_with_debug(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, dict]:
        """
        Sample actions with debug information (not JIT compiled)
        This method generates subtask descriptions and returns debug info
        """
        debug_info = {}
        
        # Split RNG for subtask generation and action sampling
        subtask_rng, action_rng = jax.random.split(rng, 2)
        
        # Generate subtask description for debugging
        try:
            observation_processed = _model.preprocess_observation(None, observation, train=False)
            
            subtask_tokens, subtask_log_probs = self.generate_subtask_description(
                rng=subtask_rng,
                observation=observation_processed,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            
            # Store raw tokens for external decoding
            debug_info.update({
                'debug_subtask_tokens': subtask_tokens,
                'debug_subtask_log_probs': subtask_log_probs
            })
            
            print(f"[DEBUG] Generated subtask tokens shape: {subtask_tokens.shape}")
            print(f"[DEBUG] Log probs shape: {subtask_log_probs.shape}")
            
        except Exception as e:
            print(f"[DEBUG] Error generating subtask description: {e}")
            import traceback
            traceback.print_exc()
            debug_info['debug_subtask_error'] = str(e)
            action_rng = rng  # Use original RNG if subtask generation fails
        
        # Sample actions using the regular method
        actions = self.sample_actions(
            rng=action_rng,
            observation=observation,
            num_steps=num_steps,
            noise=noise,
            return_debug_info=False  # Always False since this method handles debug
        )
        
        return actions, debug_info

    @at.typecheck
    def generate_subtask_description(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        subtask_prompt: str = "Next subtask:",
    ) -> tuple[at.Int[at.Array, "b new_tokens"], at.Float[at.Array, "b new_tokens"]]:
        """
        生成描述下一个子任务的文本token序列
        
        Args:
            rng: 随机数生成器
            observation: 当前观察
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            subtask_prompt: 引导生成的提示文本
            
        Returns:
            generated_tokens: 生成的token序列 [b, new_tokens]
            log_probs: 对应的对数概率 [b, new_tokens]
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        
        # 1. 创建subtask提示的tokenizer
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer(max_len=48)
        
        # 2. 编码subtask提示
        subtask_tokens, subtask_mask = tokenizer.tokenize(subtask_prompt)
        # 获取实际的序列长度
        seq_len = subtask_tokens.shape[0]
        subtask_tokens = jnp.broadcast_to(subtask_tokens[None], (batch_size, seq_len))
        subtask_mask = jnp.broadcast_to(subtask_mask[None], (batch_size, seq_len))
        
        # 3. 构建初始序列（观察 + subtask提示）
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        # 添加subtask提示tokens
        subtask_embedded = self.PaliGemma.llm(subtask_tokens, method="embed")
        full_tokens = jnp.concatenate([prefix_tokens, subtask_embedded], axis=1)
        full_mask = jnp.concatenate([prefix_mask, subtask_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, 
                                       jnp.array([False] * subtask_tokens.shape[1])], axis=0)
        
        # 4. 自回归生成循环 - 使用无KV缓存的简单方法
        generated_tokens = []
        generated_log_probs = []
        current_tokens = full_tokens
        current_mask = full_mask
        current_ar_mask = full_ar_mask
        
        for step in range(max_new_tokens):
            # 计算注意力掩码和位置
            attn_mask = make_attn_mask(current_mask, current_ar_mask)
            positions = jnp.cumsum(current_mask, axis=1) - 1
            
            # 前向传播（只使用VLM专家）- 不使用KV缓存
            (vlm_out, _), _ = self.PaliGemma.llm(
                [current_tokens, None],  # 只使用VLM专家
                mask=attn_mask,
                positions=positions,
                kv_cache=None,  # 不使用KV缓存以避免形状不匹配
                adarms_cond=[None, None]
            )
            
            # 获取最后一个token的logits
            last_hidden = vlm_out[:, -1]  # [b, d]
            # 使用Gemma模块的decode方法
            logits = self.PaliGemma.llm(last_hidden, method="decode")  # [b, vocab_size]
            
            # 应用温度缩放
            logits = logits / temperature
            
            # Top-p (nucleus) 采样
            if top_p < 1.0:
                logits = self._apply_top_p_mask(logits, top_p)
            
            # 采样下一个token
            next_token_rng, rng = jax.random.split(rng)
            probs = jax.nn.softmax(logits, axis=-1)
            next_tokens = jax.random.categorical(next_token_rng, logits, axis=-1)  # [b,]
            
            # 计算对数概率
            next_log_probs = jnp.log(jnp.take_along_axis(probs, next_tokens[:, None], axis=-1)).squeeze(-1)
            
            # 检查是否遇到EOS token
            eos_token_id = self._get_eos_token_id(tokenizer)
            is_eos = (next_tokens == eos_token_id)
            
            # 保存生成结果
            generated_tokens.append(next_tokens)
            generated_log_probs.append(next_log_probs)
            
            # JAX兼容的早期停止检查：检查是否所有序列都生成了EOS
            # 使用 jnp.all 但不在条件语句中使用
            all_eos = jnp.all(is_eos)
            
            # 为下一步准备输入
            next_embedded = self.PaliGemma.llm(next_tokens[:, None], method="embed")
            current_tokens = jnp.concatenate([current_tokens, next_embedded], axis=1)
            current_mask = jnp.concatenate([current_mask, 
                                           jnp.ones((batch_size, 1), dtype=jnp.bool_)], axis=1)
            current_ar_mask = jnp.concatenate([current_ar_mask, jnp.array([False])], axis=0)
            
            # 注意：在JAX JIT编译环境中，我们不能基于数据相关的条件提前退出循环
            # 所以我们总是生成完整的max_new_tokens序列，后续可以基于EOS标记进行截断
        
        # 转换为数组格式
        if generated_tokens:
            generated_tokens = jnp.stack(generated_tokens, axis=1)  # [b, new_tokens]
            generated_log_probs = jnp.stack(generated_log_probs, axis=1)  # [b, new_tokens]
            
            # 后处理：基于EOS token截断序列
            eos_token_id = self._get_eos_token_id(tokenizer)
            eos_positions = jnp.where(generated_tokens == eos_token_id, 
                                     jnp.arange(generated_tokens.shape[1])[None, :], 
                                     generated_tokens.shape[1])
            # 找到每个序列中第一个EOS的位置
            first_eos_pos = jnp.min(eos_positions, axis=1)
            
            # 创建掩码来截断EOS之后的tokens
            position_mask = jnp.arange(generated_tokens.shape[1])[None, :] <= first_eos_pos[:, None]
            generated_tokens = jnp.where(position_mask, generated_tokens, 0)  # 将EOS后的token设为0(padding)
            
        else:
            generated_tokens = jnp.zeros((batch_size, 0), dtype=jnp.int32)
            generated_log_probs = jnp.zeros((batch_size, 0), dtype=jnp.float32)
        
        return generated_tokens, generated_log_probs

    def _apply_top_p_mask(self, logits: at.Float[at.Array, "b vocab"], top_p: float) -> at.Float[at.Array, "b vocab"]:
        """应用top-p采样掩码"""
        sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]  # 降序排列
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        
        # 创建掩码：保留累积概率 <= top_p 的tokens
        mask = cumulative_probs <= top_p
        mask = mask.at[:, 0].set(True)  # 至少保留概率最高的token
        
        # 将掩码应用回原始顺序
        original_mask = jnp.zeros_like(logits, dtype=bool)
        original_mask = original_mask.at[jnp.arange(logits.shape[0])[:, None], sorted_indices].set(mask)
        
        # 将不满足条件的logits设为负无穷
        return jnp.where(original_mask, logits, -jnp.inf)

    def _get_eos_token_id(self, tokenizer) -> int:
        """获取EOS token的ID"""
        # 使用sentencepiece的EOS token ID
        return tokenizer._tokenizer.eos_id()

    def decode_tokens(self, tokens: at.Int[at.Array, "b t"]) -> list[str]:
        """将token序列解码为文本
        
        使用JAX兼容的方式进行token解码
        """
        from openpi.models.tokenizer import PaligemmaTokenizer
        
        # 首先尝试直接解码（如果不在JIT中）
        try:
            tokenizer = PaligemmaTokenizer()
            batch_size = tokens.shape[0]
            decoded_texts = []
            
            # 将JAX数组转换为Python列表进行处理
            tokens_list = tokens.tolist()
            
            for i in range(batch_size):
                # 移除padding tokens (通常是0)
                valid_tokens = [t for t in tokens_list[i] if t > 0]
                
                # 解码为文本
                if (valid_tokens):
                    try:
                        text = tokenizer._tokenizer.decode(valid_tokens)
                        decoded_texts.append(text.strip())
                    except Exception as e:
                        decoded_texts.append(f"decode_error_{i}")
                else:
                    decoded_texts.append("")
            
            return decoded_texts
            
        except Exception as e:
            # 如果直接解码失败（可能在JIT中），使用fallback
            print(f"[DEBUG] Direct decode failed: {e}, using fallback")
            
            # 使用JAX operations创建简单的字符串表示
            batch_size = tokens.shape[0]
            seq_len = tokens.shape[1]
            
            # 计算每个batch中有效token的数量
            valid_counts = jnp.sum(tokens > 0, axis=1)
            
            # 创建fallback文本列表
            fallback_texts = []
            for i in range(batch_size):
                count = int(valid_counts[i])
                fallback_texts.append(f"subtask_{i}_tokens_{count}")
            
            return fallback_texts
