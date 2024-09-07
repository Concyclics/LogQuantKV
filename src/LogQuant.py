from transformers import CacheConfig, QuantizedCacheConfig, DynamicCache
import torch

import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version

from quanto import AffineQuantizer, MaxOptimizer, qint2, qint4

@dataclass
class KiViSinkQuantizedCacheConfig(CacheConfig):
    """
    Configuration class for streaming quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        window_length (`Optional[int]`, *optional*, defaults to 64):
            Length of the window cache for StreamingQuantizedCache to store the full precision states. The Total full precision states will be 2 * window_length.
            Defaults to 64.
        sink_length (`Optional[int]`, *optional*, defaults to 4):
            Length of the sink cache which will always be stored in original presicion.
            Defaults to 4.
        compute_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The defualt dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to perform computations, should be same as the model's device.
    """

    def __init__(
        self,
        backend: str = "quanto",
        nbits: Optional[int] = 4,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 64,
        window_length: Optional[int] = 64,
        sink_length: Optional[int] = 2,
        compute_dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[str] = "cpu",
    ):
        self.backend = backend
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.window_length = window_length
        self.sink_length = sink_length
        self.compute_dtype = compute_dtype
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits",
                    correct_value="2 or 4 or 8",
                    found_value=self.nbits,
                ),
            )
        if self.q_group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.q_group_size,
                ),
            )
        if self.window_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="window_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )
        
        if self.sink_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="sink_length",
                    correct_value="a positive integer",
                    found_value=self.sink_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )

class KiViSinkQuantizedCache(DynamicCache):
    """
    A quantizer cache using StreamingLLM strategy to save full precision states for a fixed window length and quantize the rest of the states.
    The Total full precision states has a maximum capacity of [2 * window_length + sink_length].

    The cache has two types of storage, one for original precision and one for the quantized cache. A `window length` is set as half of a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - 2 * window_length - sink_length, head_dim]`
    """

    def __init__(self, cache_config: KiViSinkQuantizedCacheConfig) -> None:
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []
        self.sink_key_cache: List[torch.Tensor] = []
        self.sink_value_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.window_length = cache_config.window_length
        self.sink_length = cache_config.sink_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.compute_dtype = cache_config.compute_dtype
        self.device = cache_config.device

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self.sink_key_cache.append(key_states[..., :self.sink_length, :].contiguous())
            self.sink_value_cache.append(value_states[..., :self.sink_length, :].contiguous())
            self._quantized_key_cache.append(self._quantize(key_states[..., self.sink_length:, :].contiguous(), axis=self.axis_key))
            self._quantized_value_cache.append(self._quantize(value_states[..., self.sink_length:, :].contiguous(), axis=self.axis_value))
            self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))

            keys_to_return, values_to_return = key_states, value_states
        else:
            dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            keys_to_return = [self.sink_key_cache[layer_idx], dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [self.sink_value_cache[layer_idx], dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.window_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(
                    keys_to_return[..., self.sink_length:, :].contiguous(),
                    axis=self.axis_key,
                )
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return[..., self.sink_length:, :].contiguous(),
                    axis=self.axis_value,
                )
                self.key_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")


class QuantoKiViSinkQuantizedCache(KiViSinkQuantizedCache):
    """
    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`,):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        quanto_version = version.parse(importlib.metadata.version("quanto"))
        if quanto_version < version.parse("0.2.0"):
            raise ImportError(
                f"You need quanto package version to be greater or equal than 0.2.0 to use `QuantoQuantizedCache`. Detected version {quanto_version}. "
                f"Please upgrade quanto with `pip install -U quanto`"
            )

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        scale, zeropoint = self.optimizer(tensor, self.qtype.bits, axis, self.q_group_size)
        qtensor = AffineQuantizer.apply(tensor, self.qtype, axis, self.q_group_size, scale, zeropoint)
        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()

@dataclass
class StreamingQuantizedCacheConfig(CacheConfig):
    """
    Configuration class for streaming quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        window_length (`Optional[int]`, *optional*, defaults to 64):
            Length of the window cache for StreamingQuantizedCache to store the full precision states. The Total full precision states will be 2 * window_length.
            Defaults to 64.
        sink_length (`Optional[int]`, *optional*, defaults to 4):
            Length of the sink cache which will always be stored in original presicion.
            Defaults to 4.
        compute_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The defualt dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to perform computations, should be same as the model's device.
    """

    def __init__(
        self,
        backend: str = "quanto",
        nbits: Optional[int] = 4,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 64,
        window_length: Optional[int] = 64,
        sink_length: Optional[int] = 4,
        compute_dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[str] = "cpu",
    ):
        self.backend = backend
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.window_length = window_length
        self.sink_length = sink_length
        self.compute_dtype = compute_dtype
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits",
                    correct_value="2 or 4 or 8",
                    found_value=self.nbits,
                ),
            )
        if self.q_group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.q_group_size,
                ),
            )
        if self.window_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="window_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )
        
        if self.sink_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="sink_length",
                    correct_value="a positive integer",
                    found_value=self.sink_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )

class StreamingQuantizedCache(DynamicCache):
    """
    A quantizer cache using StreamingLLM strategy to save full precision states for a fixed window length and quantize the rest of the states.
    The Total full precision states has a maximum capacity of [2 * window_length + sink_length].

    The cache has two types of storage, one for original precision and one for the quantized cache. A `window length` is set as half of a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - 2 * window_length - sink_length, head_dim]`
    """

    def __init__(self, cache_config: StreamingQuantizedCacheConfig) -> None:
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []
        self.sink_key_cache: List[torch.Tensor] = []
        self.sink_value_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.window_length = cache_config.window_length
        self.sink_length = cache_config.sink_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.compute_dtype = cache_config.compute_dtype
        self.device = cache_config.device

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            if key_states.shape[-2] > self.sink_length + self.window_length:
                # Store the sink states as full precision
                self.sink_key_cache.append(key_states[..., :self.sink_length, :].contiguous())
                self.sink_value_cache.append(value_states[..., :self.sink_length, :].contiguous())
                self.key_cache.append(key_states[..., -self.window_length:, :].contiguous())
                self.value_cache.append(value_states[..., -self.window_length:, :].contiguous())
                self._quantized_key_cache.append(self._quantize(key_states[..., self.sink_length:-self.window_length, :].contiguous(), axis=self.axis_key))
                self._quantized_value_cache.append(self._quantize(value_states[..., self.sink_length:-self.window_length, :].contiguous(), axis=self.axis_value))
            else:
                self._quantized_key_cache.append(None)
                self._quantized_value_cache.append(None)
                if key_states.shape[-2] >= self.sink_length:
                    self.sink_key_cache.append(key_states[..., :self.sink_length, :].contiguous())
                    self.sink_value_cache.append(value_states[..., :self.sink_length, :].contiguous())
                    self.key_cache.append(key_states[..., self.sink_length:, :].contiguous())
                    self.value_cache.append(value_states[..., self.sink_length:, :].contiguous())
                else:
                    self.sink_key_cache.append(key_states.contiguous())
                    self.sink_value_cache.append(value_states.contiguous())
                    self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                    self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            keys_to_return, values_to_return = key_states, value_states
        else:
            if self._quantized_key_cache[layer_idx] is not None:
                dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
                dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            else:
                dequant_key = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                dequant_value = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            keys_to_return = [self.sink_key_cache[layer_idx], dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [self.sink_value_cache[layer_idx], dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= 2 * self.window_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(
                    keys_to_return[..., self.sink_length:-self.window_length, :].contiguous(),
                    axis=self.axis_key,
                )
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return[..., self.sink_length:-self.window_length, :].contiguous(),
                    axis=self.axis_value,
                )
                self.key_cache[layer_idx] = keys_to_return[..., -self.window_length:, :].contiguous()
                self.value_cache[layer_idx] = values_to_return[..., -self.window_length:, :].contiguous()
            else:
                if self.sink_key_cache[layer_idx].shape[-2] < self.sink_length:
                    self.sink_key_cache[layer_idx] = torch.cat([self.sink_key_cache[layer_idx], key_states], dim=-2)
                    self.sink_value_cache[layer_idx] = torch.cat([self.sink_value_cache[layer_idx], value_states], dim=-2)
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")


class QuantoStreamingQuantizedCache(StreamingQuantizedCache):
    """
    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`,):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        quanto_version = version.parse(importlib.metadata.version("quanto"))
        if quanto_version < version.parse("0.2.0"):
            raise ImportError(
                f"You need quanto package version to be greater or equal than 0.2.0 to use `QuantoQuantizedCache`. Detected version {quanto_version}. "
                f"Please upgrade quanto with `pip install -U quanto`"
            )

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        scale, zeropoint = self.optimizer(tensor, self.qtype.bits, axis, self.q_group_size)
        qtensor = AffineQuantizer.apply(tensor, self.qtype, axis, self.q_group_size, scale, zeropoint)
        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()
    
@dataclass
class PartialStreamingQuantizedCacheConfig(CacheConfig):
    """
    Configuration class for streaming quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        window_length (`Optional[int]`, *optional*, defaults to 64):
            Length of the window cache for StreamingQuantizedCache to store the full precision states. The Total full precision states of Key will be 2 * window_length.
            Length of the window cache for StreamingQuantizedCache to store the full precision states. The Total full precision states of Value will be window_length.
            Defaults to 64.
        sink_length (`Optional[int]`, *optional*, defaults to 4):
            Length of the sink cache which will always be stored in original presicion.
            Defaults to 4.
        compute_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The defualt dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to perform computations, should be same as the model's device.
    """

    def __init__(
        self,
        backend: str = "quanto",
        nbits: Optional[int] = 4,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 64,
        window_length: Optional[int] = 64,
        sink_length: Optional[int] = 4,
        compute_dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[str] = "cpu",
    ):
        self.backend = backend
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.window_length = window_length
        self.sink_length = sink_length
        self.compute_dtype = compute_dtype
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits",
                    correct_value="2 or 4 or 8",
                    found_value=self.nbits,
                ),
            )
        if self.q_group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.q_group_size,
                ),
            )
        if self.window_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="window_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )
        
        if self.sink_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="sink_length",
                    correct_value="a positive integer",
                    found_value=self.sink_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )

class PartialStreamingQuantizedCache(DynamicCache):
    """
    A quantizer cache using StreamingLLM strategy to save full precision states for a fixed window length and quantize the rest of the states.
    The Key and Value cache are stored with different lengths. 
    The Total full precision states of Key will be [2 * window_length + sink_length].
    The Total full precision states of Value will be [window_length + sink_length].

    The cache has two types of storage, one for original precision and one for the quantized cache. A `window length` is set as half of a maximum capacity for the
    original precision key cache and a maximum capacity for the original precision value cache. When the length goes beyond maximum capacity, 
    the original precision cache is discarded and moved into the quantized cache.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - 2 * window_length - sink_length, head_dim]`
    """

    def __init__(self, cache_config: PartialStreamingQuantizedCacheConfig) -> None:
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []
        self.sink_key_cache: List[torch.Tensor] = []
        self.sink_value_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.window_length = cache_config.window_length
        self.sink_length = cache_config.sink_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.compute_dtype = cache_config.compute_dtype
        self.device = cache_config.device

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            if key_states.shape[-2] >= self.sink_length + self.window_length:
                self._quantized_key_cache.append(self._quantize(key_states[..., self.sink_length:-self.window_length, :].contiguous(), axis=self.axis_key))
                self._quantized_value_cache.append(self._quantize(value_states[..., self.sink_length:, :].contiguous(), axis=self.axis_value))
                # Store the sink states as full precision
                self.sink_key_cache.append(key_states[..., :self.sink_length, :].contiguous())
                self.sink_value_cache.append(value_states[..., :self.sink_length, :].contiguous())
                self.key_cache.append(key_states[..., -self.window_length:, :].contiguous())
                self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            else:
                self._quantized_key_cache.append(None)
                self._quantized_value_cache.append(None)
                if key_states.shape[-2] >= self.sink_length:
                    self.sink_key_cache.append(key_states[..., :self.sink_length, :].contiguous())
                    self.sink_value_cache.append(value_states[..., :self.sink_length, :].contiguous())
                    self.key_cache.append(key_states[..., self.sink_length:, :].contiguous())
                    self.value_cache.append(value_states[..., self.sink_length:, :].contiguous())
                else:
                    self.sink_key_cache.append(key_states.contiguous())
                    self.sink_value_cache.append(value_states.contiguous())
                    self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                    self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                    
            keys_to_return, values_to_return = key_states, value_states
            
        else:
            if self._quantized_key_cache[layer_idx] is not None:
                dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            else:
                dequant_key = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            if self._quantized_value_cache[layer_idx] is not None:
                dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            else:
                dequant_value = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            keys_to_return = [self.sink_key_cache[layer_idx], dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [self.sink_value_cache[layer_idx], dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= 2 * self.window_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(
                    keys_to_return[..., self.sink_length:-self.window_length, :].contiguous(),
                    axis=self.axis_key,
                )
                self.key_cache[layer_idx] = keys_to_return[..., -self.window_length:, :].contiguous()
            else:
                if self.sink_key_cache[layer_idx].shape[-2] < self.sink_length:
                    self.sink_key_cache[layer_idx] = torch.cat([self.sink_key_cache[layer_idx], key_states], dim=-2)
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)

            if (
                self.value_cache[layer_idx].dim() == 4
                and self.value_cache[layer_idx].shape[-2] + 1 >= self.window_length
            ):
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return[..., self.sink_length:, :].contiguous(),
                    axis=self.axis_value,
                )
                self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                if self.sink_key_cache[layer_idx].shape[-2] < self.sink_length:
                    self.sink_value_cache[layer_idx] = torch.cat([self.sink_value_cache[layer_idx], value_states], dim=-2)
                else:
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")


class QuantoPartialStreamingQuantizedCache(PartialStreamingQuantizedCache):
    """
    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`,):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        quanto_version = version.parse(importlib.metadata.version("quanto"))
        if quanto_version < version.parse("0.2.0"):
            raise ImportError(
                f"You need quanto package version to be greater or equal than 0.2.0 to use `QuantoQuantizedCache`. Detected version {quanto_version}. "
                f"Please upgrade quanto with `pip install -U quanto`"
            )

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        scale, zeropoint = self.optimizer(tensor, self.qtype.bits, axis, self.q_group_size)
        qtensor = AffineQuantizer.apply(tensor, self.qtype, axis, self.q_group_size, scale, zeropoint)
        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()

@dataclass
class LogQuantizedCacheConfig(CacheConfig):
    """
    Configuration class for quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        window_length (`Optional[int]`, *optional*, defaults to 64):
            Length of the window cache for LogQuant to store the full precision states. The Total full precision states will be 3 * window_length.
            Defaults to 64.
        compute_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The defualt dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to perform computations, should be same as the model's device.
    """

    def __init__(
        self,
        backend: str = "quanto",
        nbits: Optional[int] = 4,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 64,
        window_length: Optional[int] = 64,
        compute_dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[str] = "cpu",
    ):
        self.backend = backend
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.window_length = window_length
        self.compute_dtype = compute_dtype
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits",
                    correct_value="2 or 4 or 8",
                    found_value=self.nbits,
                ),
            )
        if self.q_group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.q_group_size,
                ),
            )
        if self.window_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="window_length",
                    correct_value="a positive integer",
                    found_value=self.window_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )

class LogQuantizedCache(DynamicCache):
    """
    A quantizer cache using our LogQuant strategy to save full precision states for a fixed window length and quantize the rest of the states.
    The Total full precision states has a maximum capacity of [3 * window_length].

    The cache has two types of storage, one for original precision and one for the quantized cache. A `window length` is set as one-third of a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - 3 * window_length, head_dim]`
    """

    def __init__(self, cache_config: LogQuantizedCacheConfig) -> None:
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.window_length = cache_config.window_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.compute_dtype = cache_config.compute_dtype
        self.device = cache_config.device

        self.local_slide_window_index = []
        self.full_precision_logsparse_index = []
        self.log_sparse_key_cache: List[torch.Tensor] = []
        self.log_sparse_value_cache: List[torch.Tensor] = []

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            #Prefill Phase need to filter out log sparse part index
            if key_states.shape[-2] > 1:
                local_index = list(range(key_states.shape[-2]))
                log_sparse_index = []
                tmp_index = log_sparse_index + local_index
                while len(tmp_index) >= 3 * self.window_length:
                    log_sparse_index = tmp_index[0: 2 * self.window_length: 2]
                    local_index = tmp_index[2 * self.window_length:]
                    tmp_index = log_sparse_index + local_index
                if key_states.shape[-2] < 3 * self.window_length and key_states.shape[-2] > self.window_length:
                    log_sparse_index = tmp_index[:self.window_length]
                    local_index = tmp_index[self.window_length:]
                self.local_slide_window_index = local_index
                self.full_precision_logsparse_index = log_sparse_index

            else:
                self.local_slide_window_index.append(self._seen_tokens - 1)
                if len(self.local_slide_window_index) > 2 * self.window_length:
                    if len(self.full_precision_logsparse_index) > 0:
                        self.full_precision_logsparse_index = (self.full_precision_logsparse_index + self.local_slide_window_index[0:self.window_length])[0::2]
                    else:
                        self.full_precision_logsparse_index = self.local_slide_window_index[:self.window_length]
                    self.local_slide_window_index = self.local_slide_window_index[self.window_length:]

        if len(self.key_cache) <= layer_idx:
            if self.local_slide_window_index[0] == 0:
                self._quantized_key_cache.append(None)
                self._quantized_value_cache.append(None)
            else:
                self._quantized_key_cache.append(self._quantize(key_states[..., :self.local_slide_window_index[0], :].contiguous(), axis=self.axis_key))
                self._quantized_value_cache.append(self._quantize(value_states[..., :self.local_slide_window_index[0], :].contiguous(), axis=self.axis_value))
            self.key_cache.append(key_states[..., self.local_slide_window_index, :].contiguous())
            self.value_cache.append(value_states[..., self.local_slide_window_index, :].contiguous())
            self.log_sparse_key_cache.append(key_states[..., self.full_precision_logsparse_index, :].contiguous())
            self.log_sparse_value_cache.append(value_states[..., self.full_precision_logsparse_index, :].contiguous())

            keys_to_return, values_to_return = key_states, value_states
        else:
            if self._quantized_key_cache[layer_idx] is None:
                dequant_key = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                dequant_value = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
                dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
                
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            
            if len(self.full_precision_logsparse_index) > 0:
                keys_to_return[..., self.full_precision_logsparse_index, :] = self.log_sparse_key_cache[layer_idx]
                values_to_return[..., self.full_precision_logsparse_index, :] = self.log_sparse_value_cache[layer_idx]

            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= 2 * self.window_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(
                    keys_to_return[..., :-self.window_length, :].contiguous(),
                    axis=self.axis_key)

                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return[..., :-self.window_length, :].contiguous(), 
                    axis=self.axis_value)
                if len(self.full_precision_logsparse_index) > 0:
                    self.log_sparse_key_cache[layer_idx] = torch.cat([
                        self.log_sparse_key_cache[layer_idx][..., 0::2, :],
                        self.key_cache[layer_idx][..., self.window_length%2:self.window_length:2, :]
                    ], dim=-2)

                    self.log_sparse_value_cache[layer_idx] = torch.cat([
                        self.log_sparse_value_cache[layer_idx][..., 0::2, :],
                        self.value_cache[layer_idx][..., self.window_length%2:self.window_length:2, :]
                    ], dim=-2)
                else:
                    self.log_sparse_key_cache[layer_idx] = self.key_cache[layer_idx][..., :self.window_length, :].contiguous()
                    self.log_sparse_value_cache[layer_idx] = self.value_cache[layer_idx][..., :self.window_length, :].contiguous()             
                
                self.key_cache[layer_idx] = keys_to_return[..., -self.window_length:, :].contiguous()
                self.value_cache[layer_idx] = values_to_return[..., -self.window_length:, :].contiguous()
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            
        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1
    
    def get_log_sparse_index(self, layer_idx: Optional[int] = 0) -> List[int]:
        if len(self.key_cache) <= layer_idx:
            return []
        return self.full_precision_logsparse_index
    
    def get_local_slide_window_index(self, layer_idx: Optional[int] = 0) -> List[int]:
        if len(self.key_cache) <= layer_idx:
            return []
        return self.local_slide_window_index

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")


class QuantoLogQuantizedCache(LogQuantizedCache):
    """
    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`,):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        quanto_version = version.parse(importlib.metadata.version("quanto"))
        if quanto_version < version.parse("0.2.0"):
            raise ImportError(
                f"You need quanto package version to be greater or equal than 0.2.0 to use `QuantoQuantizedCache`. Detected version {quanto_version}. "
                f"Please upgrade quanto with `pip install -U quanto`"
            )

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        scale, zeropoint = self.optimizer(tensor, self.qtype.bits, axis, self.q_group_size)
        qtensor = AffineQuantizer.apply(tensor, self.qtype, axis, self.q_group_size, scale, zeropoint)
        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()
    
@dataclass
class PartialLogQuantizedCacheConfig(CacheConfig):
    """
    Configuration class for quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        window_length (`Optional[int]`, *optional*, defaults to 64):
            Length of the window cache for LogQuant to store the full precision states. The Total full precision states will be 3 * window_length.
            Defaults to 64.
        compute_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The defualt dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to perform computations, should be same as the model's device.
    """

    def __init__(
        self,
        backend: str = "quanto",
        nbits: Optional[int] = 4,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 64,
        window_length: Optional[int] = 64,
        compute_dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[str] = "cpu",
    ):
        self.backend = backend
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.window_length = window_length
        self.compute_dtype = compute_dtype
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits",
                    correct_value="2 or 4 or 8",
                    found_value=self.nbits,
                ),
            )
        if self.q_group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.q_group_size,
                ),
            )
        if self.window_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="window_length",
                    correct_value="a positive integer",
                    found_value=self.window_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )

class PartialLogQuantizedCache(DynamicCache):
    """
    A quantizer cache using our LogQuant strategy to save full precision states for a fixed window length and quantize the rest of the states.
    The Total full precision key states has a maximum capacity of [3 * window_length].
    The Total full precision value states has a maximum capacity of [window_length].

    The cache has two types of storage, one for original precision and one for the quantized cache. A `window length` is set as one-third of a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - 3 * window_length, head_dim]`

    """

    def __init__(self, cache_config: PartialLogQuantizedCacheConfig) -> None:
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.window_length = cache_config.window_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.compute_dtype = cache_config.compute_dtype
        self.device = cache_config.device

        self.local_slide_window_index = []
        self.full_precision_logsparse_index = []
        self.log_sparse_key_cache: List[torch.Tensor] = []

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            #Prefill Phase need to filter out log sparse part index
            if key_states.shape[-2] > 1:
                local_index = list(range(key_states.shape[-2]))
                log_sparse_index = []
                tmp_index = log_sparse_index + local_index
                while len(tmp_index) >= 3 * self.window_length:
                    log_sparse_index = tmp_index[0: 2 * self.window_length: 2]
                    local_index = tmp_index[2 * self.window_length:]
                    tmp_index = log_sparse_index + local_index
                if key_states.shape[-2] < 3 * self.window_length and key_states.shape[-2] > self.window_length:
                    log_sparse_index = tmp_index[:self.window_length]
                    local_index = tmp_index[self.window_length:]
                self.local_slide_window_index = local_index
                self.full_precision_logsparse_index = log_sparse_index

                self.local_slide_window_index = local_index
                self.full_precision_logsparse_index = log_sparse_index
            else:
                self.local_slide_window_index.append(self._seen_tokens - 1)
                if len(self.local_slide_window_index) > 2 * self.window_length:
                    if len(self.full_precision_logsparse_index) > 0:
                        self.full_precision_logsparse_index = (self.full_precision_logsparse_index + self.local_slide_window_index[0:self.window_length])[0::2]
                    else:
                        self.full_precision_logsparse_index = self.local_slide_window_index[0:self.window_length:1]
                    self.local_slide_window_index = self.local_slide_window_index[self.window_length:]

        if len(self.key_cache) <= layer_idx:
            if self.local_slide_window_index[0] == 0:
                self._quantized_key_cache.append(None)
            else:
                self._quantized_key_cache.append(self._quantize(key_states[..., :self.local_slide_window_index[0], :].contiguous(), axis=self.axis_key))
            self.key_cache.append(key_states[..., self.local_slide_window_index, :].contiguous())
            self.log_sparse_key_cache.append(key_states[..., self.full_precision_logsparse_index, :].contiguous())
        
            rest = value_states.shape[-2] % self.window_length
            if rest == 0:
                self._quantized_value_cache.append(self._quantize(value_states.contiguous(), axis=self.axis_value))
                self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            else:
                if value_states.shape[-2] > rest:
                    self._quantized_value_cache.append(self._quantize(value_states[..., :-rest, :].contiguous(), axis=self.axis_value))
                else:
                    self._quantized_value_cache.append(None)
                self.value_cache.append(value_states[..., -rest:, :].contiguous())
            keys_to_return, values_to_return = key_states, value_states
        else:
            if self._quantized_key_cache[layer_idx] is None:
                dequant_key = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            if self._quantized_value_cache[layer_idx] is None:
                dequant_value = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
                
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]
            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)

            if len(self.full_precision_logsparse_index) > 0:
                keys_to_return[..., self.full_precision_logsparse_index, :] = self.log_sparse_key_cache[layer_idx]

            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= 2 * self.window_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(
                    keys_to_return[..., :-self.window_length, :].contiguous(),
                    axis=self.axis_key)
                if len(self.full_precision_logsparse_index) > 0:
                    self.log_sparse_key_cache[layer_idx] = torch.cat([
                        self.log_sparse_key_cache[layer_idx][..., 0::2, :],
                        self.key_cache[layer_idx][..., self.window_length%2:self.window_length:2, :]
                    ], dim=-2)
                else:
                    self.log_sparse_key_cache[layer_idx] = self.key_cache[layer_idx][..., :self.window_length, :].contiguous()

                self.key_cache[layer_idx] = keys_to_return[..., -self.window_length:, :].contiguous()
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)

            if (
                self.value_cache[layer_idx].dim() == 4
                and self.value_cache[layer_idx].shape[-2] + 1 >= self.window_length
            ):
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(),
                    axis=self.axis_value)
                
                self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                
        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1
    
    def get_log_sparse_index(self, layer_idx: Optional[int] = 0) -> List[int]:
        if len(self.key_cache) <= layer_idx:
            return []
        return self.full_precision_logsparse_index
    
    def get_local_slide_window_index(self, layer_idx: Optional[int] = 0) -> List[int]:
        if len(self.key_cache) <= layer_idx:
            return []
        return self.local_slide_window_index

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")


class QuantoPartialLogQuantizedCache(PartialLogQuantizedCache):
    """
    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`,):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        quanto_version = version.parse(importlib.metadata.version("quanto"))
        if quanto_version < version.parse("0.2.0"):
            raise ImportError(
                f"You need quanto package version to be greater or equal than 0.2.0 to use `QuantoQuantizedCache`. Detected version {quanto_version}. "
                f"Please upgrade quanto with `pip install -U quanto`"
            )

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        scale, zeropoint = self.optimizer(tensor, self.qtype.bits, axis, self.q_group_size)
        qtensor = AffineQuantizer.apply(tensor, self.qtype, axis, self.q_group_size, scale, zeropoint)
        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()
