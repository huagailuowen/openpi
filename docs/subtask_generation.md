# Pi0 Subtask Generation

Pi0 模型现在支持生成关于下一步子任务的文本描述，这个功能利用了模型的视觉-语言专家（VLM expert）来理解当前场景并预测合适的下一步动作描述。

## 功能概述

- **用途**: 为低级规划器提供高级指导信息
- **输入**: 多模态观察（图像、状态、文本提示）
- **输出**: 描述下一步子任务的自然语言文本
- **实现**: 基于 VLM 专家的自回归文本生成

## API 使用

### 基本用法

```python
from openpi.models import pi0, pi0_config, model
import jax.random

# 创建模型
config = pi0_config.Pi0Config(pi05=True, action_dim=7, action_horizon=4)
model = pi0.Pi0(config, {"params": jax.random.PRNGKey(42)})

# 准备观察数据
observation = model.Observation(
    images={"camera": camera_image},    # [b, h, w, c]
    image_masks={"camera": image_mask}, # [b]
    state=robot_state,                  # [b, state_dim]
    tokenized_prompt=prompt_tokens,     # [b, seq_len] (可选)
    tokenized_prompt_mask=prompt_mask   # [b, seq_len] (可选)
)

# 生成子任务描述
generated_tokens, log_probs = model.generate_subtask_description(
    rng=jax.random.PRNGKey(123),
    observation=observation,
    max_new_tokens=30,
    temperature=0.7,
    top_p=0.9,
    subtask_prompt="Next subtask:"
)

# 解码为文本
texts = model.decode_tokens(generated_tokens)
print(f"Generated subtask: {texts[0]}")
```

### 参数说明

- `max_new_tokens`: 最大生成的 token 数量（默认 50）
- `temperature`: 控制生成多样性，越高越随机（默认 0.7）
- `top_p`: Nucleus 采样参数，控制候选词汇范围（默认 0.9）
- `subtask_prompt`: 引导生成的提示文本（默认 "Next subtask:"）

## 技术实现

### 架构设计

1. **专家分离**: 只使用 VLM 专家进行文本生成，不涉及动作专家
2. **注意力机制**: 支持图像、文本和生成文本之间的交互注意力
3. **自回归生成**: 逐 token 生成，支持 KV 缓存加速
4. **采样控制**: 支持 temperature 和 top-p 采样策略

### 工作流程

```
观察输入 → 多模态嵌入 → VLM专家处理 → 自回归生成 → 文本解码
```

1. **前缀构建**: 将图像和文本观察编码为 token 序列
2. **提示添加**: 添加子任务生成的引导提示
3. **自回归生成**: 逐步生成描述下一步任务的 token
4. **文本解码**: 将生成的 token 转换为可读文本

## 应用场景

### 1. 分层规划
```python
# 高级规划器生成子任务描述
subtask_description = model.generate_subtask_description(
    observation=current_obs,
    subtask_prompt="To complete the task, next I should:"
)

# 低级规划器根据描述执行具体动作
actions = low_level_planner.plan(
    observation=current_obs,
    subtask_description=subtask_description
)
```

### 2. 人机交互
```python
# 为用户解释机器人的下一步计划
explanation = model.generate_subtask_description(
    observation=current_obs,
    subtask_prompt="The robot is planning to:"
)
print(f"Robot says: {explanation}")
```

### 3. 调试和可解释性
```python
# 生成模型对当前情况的理解
reasoning = model.generate_subtask_description(
    observation=current_obs,
    subtask_prompt="Current situation analysis:"
)
logger.info(f"Model reasoning: {reasoning}")
```

## 性能考虑

- **计算开销**: 自回归生成比直接动作预测慢，适合非实时场景
- **内存使用**: KV 缓存机制减少重复计算，但会增加内存使用
- **生成质量**: 依赖于 VLM 专家的预训练质量和微调数据

## 局限性

1. **生成长度**: 受模型最大序列长度限制
2. **一致性**: 生成的文本可能与实际执行的动作不完全一致
3. **语言模型限制**: 受制于底层 tokenizer 和语言模型的能力
4. **计算资源**: 需要额外的计算资源进行文本生成

## 扩展方向

1. **多轮对话**: 支持连续的子任务生成和用户交互
2. **条件生成**: 基于特定条件（如安全约束）生成描述
3. **多语言支持**: 扩展到多种自然语言的生成
4. **结构化输出**: 生成结构化的任务描述（JSON、YAML等）
