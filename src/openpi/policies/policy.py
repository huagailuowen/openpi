from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def decode_subtask_tokens(tokens: jnp.ndarray) -> list[str]:
    """
    解码subtask tokens为可读文本
    这个函数在JIT编译之外调用，避免JAX的限制
    """
    from openpi.models.tokenizer import PaligemmaTokenizer
    
    try:
        tokenizer = PaligemmaTokenizer()
        batch_size = tokens.shape[0]
        decoded_texts = []
        
        for i in range(batch_size):
            # 转换为numpy并移除padding tokens
            tokens_np = np.array(tokens[i])
            valid_tokens = [int(t) for t in tokens_np if t > 0]
            
            # 解码为文本
            if valid_tokens:
                text = tokenizer._tokenizer.decode(valid_tokens)
                decoded_texts.append(text.strip())
            else:
                decoded_texts.append("")
        
        return decoded_texts
    except Exception as e:
        logging.warning(f"Failed to decode subtask tokens: {e}")
        batch_size = tokens.shape[0]
        return [f"decode_error_{i}" for i in range(batch_size)]


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # Enable debug info for Pi0 models (including Pi0.5)
        # Check if model is Pi0 or Pi0.5 by looking for specific attributes and methods
        has_generate_method = hasattr(self._model, 'generate_subtask_description')
        has_pi0_attrs = (hasattr(self._model, 'pi05') and hasattr(self._model, 'PaliGemma') and 
                        hasattr(self._model, 'action_in_proj'))
        has_debug_method = hasattr(self._model, 'sample_actions_with_debug')
        
        is_pi0_model = has_generate_method or has_pi0_attrs

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        # switch !!!!!!!!!!!!!!!!!!!!!!
        is_debug = True
        
        if is_debug and is_pi0_model and not self._is_pytorch_model and has_debug_method:
            # Use the debug method for Pi0 models
            model_type = "Pi0.5" if (hasattr(self._model, 'pi05') and self._model.pi05) else "Pi0"
            print(f"[DEBUG] Detected {model_type} model, using debug sampling method")
            
            # Use the non-JIT debug method
            actions, debug_info = self._model.sample_actions_with_debug(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            )
            
        elif is_pi0_model and not self._is_pytorch_model:
            # Fallback to regular method if debug method not available
            sample_kwargs["return_debug_info"] = True
            model_type = "Pi0.5" if (hasattr(self._model, 'pi05') and self._model.pi05) else "Pi0"
            print(f"[DEBUG] Detected {model_type} model, enabling debug mode (fallback)")
            
            sample_result = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            if isinstance(sample_result, tuple):
                actions, debug_info = sample_result
            else:
                actions = sample_result
                debug_info = {}
                
        elif not self._is_pytorch_model:
            print(f"[DEBUG] Not a Pi0 model, debug mode disabled")
            print(f"[DEBUG] Model type: {type(self._model).__name__}")
            sample_result = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            actions = sample_result
            debug_info = {}
        else:
            # PyTorch model
            sample_result = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            actions = sample_result
            debug_info = {}
        
        # Decode subtask tokens if available in debug_info
        if 'debug_subtask_tokens' in debug_info:
            try:
                subtask_tokens = debug_info['debug_subtask_tokens']
                decoded_texts = decode_subtask_tokens(subtask_tokens)
                debug_info['debug_subtask_text'] = decoded_texts
                
                # Log the decoded subtask description
                logger = logging.getLogger(__name__)
                logger.info(f"Generated subtask description: {decoded_texts}")
                print(f"[DEBUG] Generated subtask description: {decoded_texts}")
                
            except Exception as e:
                print(f"[DEBUG] Error decoding subtask tokens: {e}")
                debug_info['debug_decode_error'] = str(e)
        
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            # Process only tensor/array outputs, not debug strings
            tensor_outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            # Process only tensor/array outputs, not debug strings  
            tensor_outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # Add debug info after tensor processing to avoid indexing issues
        if debug_info:
            tensor_outputs.update(debug_info)
            
        tensor_outputs = self._output_transform(tensor_outputs)
        tensor_outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return tensor_outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
