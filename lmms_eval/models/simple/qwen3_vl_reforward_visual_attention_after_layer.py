import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType

import torch
from tqdm import tqdm
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VL_Base


def _compute_visual_attention_ratios(
    attn_weights: torch.Tensor, visual_token_indices_by_sample: list[list[int]]
) -> list[float]:
    ratios = []
    for sample_idx, visual_token_indices in enumerate(visual_token_indices_by_sample):
        if not visual_token_indices:
            ratios.append(0.0)
            continue

        index_tensor = torch.tensor(visual_token_indices, device=attn_weights.device, dtype=torch.long)
        index_tensor = index_tensor[index_tensor < attn_weights.shape[-1]]
        if index_tensor.numel() == 0:
            ratios.append(0.0)
            continue

        per_head_visual_mass = attn_weights[sample_idx, :, -1, index_tensor].sum(dim=-1)
        ratios.append(float(per_head_visual_mass.mean().item()))
    return ratios


class VisualAttentionTopKReForwardHistoryOnlyPatcher:
    def __init__(self, model, after_layer: int, candidate_start: int, candidate_end: int, top_k: int):
        self.language_model = model.model.language_model
        self.after_layer_idx = after_layer - 1
        self.candidate_start_idx = candidate_start - 1
        self.candidate_end_idx = candidate_end - 1
        self.top_k = top_k
        self.window_size = top_k
        self.original_forward = self.language_model.forward
        self.visual_token_indices_by_sample = []
        self.cached_visual_token_indices_by_sample = []
        self.last_selected_layers = []
        self.last_selected_window_means = []
        self.selected_window_history_by_sample = []
        self._selection_history_step_idx = 0

    def _reset_selected_window_history(self, batch_size: int):
        self.selected_window_history_by_sample = [[] for _ in range(batch_size)]
        self._selection_history_step_idx = 0

    def _record_selected_windows(
        self,
        selected_layer_indices_by_sample: list[list[int]],
        selected_window_means: list[float],
        cache_position: torch.Tensor,
    ):
        batch_size = len(selected_layer_indices_by_sample)
        if len(self.selected_window_history_by_sample) != batch_size:
            self._reset_selected_window_history(batch_size)

        cache_positions = cache_position.detach().cpu().tolist()
        if not isinstance(cache_positions, list):
            cache_positions = [int(cache_positions)]
        cache_position_value = int(cache_positions[-1])

        for sample_idx, selected_layer_indices in enumerate(selected_layer_indices_by_sample):
            layers = [layer_idx + 1 for layer_idx in selected_layer_indices]
            if layers:
                window = [layers[0], layers[-1]]
            else:
                window = []
            #print(window)
            self.selected_window_history_by_sample[sample_idx].append(
                {
                    "step": self._selection_history_step_idx,
                    "cache_position": cache_position_value,
                    "window": window,
                    "layers": layers,
                    "window_mean": float(selected_window_means[sample_idx]),
                }
            )

        self._selection_history_step_idx += 1

    def _capture_visual_token_indices(
        self, visual_pos_masks: torch.Tensor | None, batch_size: int, is_decode_step: bool
    ):
        if visual_pos_masks is None:
            if is_decode_step and len(self.cached_visual_token_indices_by_sample) == batch_size:
                self.visual_token_indices_by_sample = [
                    visual_token_indices.copy()
                    for visual_token_indices in self.cached_visual_token_indices_by_sample
                ]
                return
            self.visual_token_indices_by_sample = [[] for _ in range(batch_size)]
            if not is_decode_step:
                self.cached_visual_token_indices_by_sample = []
            return

        visual_token_indices_by_sample = [
            torch.nonzero(sample_visual_pos_mask, as_tuple=False).flatten().tolist()
            for sample_visual_pos_mask in visual_pos_masks
        ]
        self.visual_token_indices_by_sample = [visual_token_indices.copy() for visual_token_indices in visual_token_indices_by_sample]
        self.cached_visual_token_indices_by_sample = [visual_token_indices.copy() for visual_token_indices in visual_token_indices_by_sample]

    def _is_decode_step(self, inputs_embeds: torch.Tensor, past_key_values) -> bool:
        if past_key_values is None:
            return False
        return past_key_values.get_seq_length() > 0 and inputs_embeds.shape[1] == 1

    def _forward_layer_with_attention(
        self,
        layer,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        text_position_ids: torch.Tensor,
        past_key_values,
        cache_position: torch.Tensor,
        position_embeddings,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, attn_weights = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights

    def _select_top_k_layers(self, candidate_ratios: list[tuple[int, float]]) -> list[int]:
        if not candidate_ratios:
            return []

        window_size = min(self.window_size, len(candidate_ratios))
        window_sum = sum(ratio for _layer_idx, ratio in candidate_ratios[:window_size])
        best_sum = window_sum
        best_start = 0

        for start in range(1, len(candidate_ratios) - window_size + 1):
            window_sum += candidate_ratios[start + window_size - 1][1] - candidate_ratios[start - 1][1]
            if window_sum > best_sum:
                best_sum = window_sum
                best_start = start

        return [layer_idx for layer_idx, _ratio in candidate_ratios[best_start : best_start + window_size]]

    def _compute_window_mean(
        self,
        candidate_ratios: list[tuple[int, float]],
        selected_layer_indices: list[int],
    ) -> float:
        if not selected_layer_indices:
            return 0.0

        ratio_by_layer = dict(candidate_ratios)
        return sum(ratio_by_layer[layer_idx] for layer_idx in selected_layer_indices) / len(selected_layer_indices)

    def _clone_cache(self, past_key_values, sample_idx: int | None = None) -> DynamicCache:
        temp_cache = DynamicCache(config=self.language_model.config)
        if past_key_values is None:
            return temp_cache

        for layer_idx, layer_cache in enumerate(past_key_values.layers):
            if layer_cache is None or not getattr(layer_cache, "is_initialized", False):
                continue
            if layer_cache.get_seq_length() == 0:
                continue

            keys = layer_cache.keys if sample_idx is None else layer_cache.keys[sample_idx : sample_idx + 1]
            values = layer_cache.values if sample_idx is None else layer_cache.values[sample_idx : sample_idx + 1]
            temp_cache.update(keys.clone(), values.clone(), layer_idx)

        return temp_cache

    def _get_or_create_probe_cache(self, past_key_values, config) -> DynamicCache:
        if past_key_values is None:
            return DynamicCache(config=config)
        return past_key_values

    def _build_history_only_cache(self, past_key_values, layer_idx: int, sample_idx: int, config) -> DynamicCache:
        temp_cache = DynamicCache(config=config)
        if past_key_values is None or len(past_key_values.layers) <= layer_idx:
            return temp_cache

        layer_cache = past_key_values.layers[layer_idx]
        if layer_cache is None or not getattr(layer_cache, "is_initialized", False):
            return temp_cache
        if layer_cache.get_seq_length() <= 1:
            return temp_cache

        history_keys = layer_cache.keys[sample_idx : sample_idx + 1, :, :-1, :]
        history_values = layer_cache.values[sample_idx : sample_idx + 1, :, :-1, :]
        if history_keys.shape[-2] == 0:
            return temp_cache

        temp_cache.update(history_keys, history_values, 0)
        return temp_cache

    def _slice_sample_attention_mask(self, attention_mask: torch.Tensor | None, sample_idx: int):
        if attention_mask is None:
            return None
        if attention_mask.ndim in (2, 4):
            return attention_mask[sample_idx : sample_idx + 1]
        raise ValueError(
            "qwen3_vl_reforward_visual_attention_after_layer only supports 2D or 4D attention masks for batched reforward"
        )

    def _slice_sample_tensor(self, tensor, sample_idx: int):
        if tensor is None:
            return None
        return tensor[sample_idx : sample_idx + 1]

    def _slice_sample_deepstack_visual_embeds(self, deepstack_visual_embeds, visual_pos_masks, sample_idx: int):
        if deepstack_visual_embeds is None or visual_pos_masks is None:
            return deepstack_visual_embeds

        visual_counts = visual_pos_masks.to(dtype=torch.long).sum(dim=-1).tolist()
        start = sum(visual_counts[:sample_idx])
        end = start + visual_counts[sample_idx]
        return [visual_embeds[start:end] for visual_embeds in deepstack_visual_embeds]

    def _split_position_ids(self, position_ids: torch.Tensor):
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids
        return text_position_ids, rotary_position_ids

    def _slice_sequence_tensor(self, tensor, end: int):
        if tensor is None:
            return None
        return tensor[..., :end]

    def _should_split_prefill_for_first_token(self, inputs_embeds: torch.Tensor, past_key_values, use_cache) -> bool:
        if not use_cache or inputs_embeds.shape[1] <= 1:
            return False
        if past_key_values is None:
            return True
        return past_key_values.get_seq_length() == 0

    def _run_probe_decode(
        self,
        language_model_self,
        inputs_embeds: torch.Tensor,
        input_attention_mask,
        text_position_ids: torch.Tensor,
        rotary_position_ids: torch.Tensor,
        past_key_values,
        cache_position: torch.Tensor,
        visual_pos_masks,
        deepstack_visual_embeds,
        kwargs: dict,
    ):
        probe_cache = self._get_or_create_probe_cache(past_key_values, language_model_self.config)
        attention_mask = create_causal_mask(
            config=language_model_self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=input_attention_mask,
            cache_position=cache_position,
            past_key_values=probe_cache,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = language_model_self.rotary_emb(hidden_states, rotary_position_ids)
        batch_size = hidden_states.shape[0]
        candidate_ratios_by_sample = [[] for _ in range(batch_size)]
        after_layer_hidden_states = None

        for layer_idx, decoder_layer in enumerate(language_model_self.layers):
            hidden_states, attn_weights = self._forward_layer_with_attention(
                decoder_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                text_position_ids=text_position_ids,
                past_key_values=probe_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = language_model_self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

            if self.candidate_start_idx <= layer_idx <= self.candidate_end_idx:
                candidate_layer_ratios = _compute_visual_attention_ratios(attn_weights, self.visual_token_indices_by_sample)
                for sample_idx, ratio in enumerate(candidate_layer_ratios):
                    candidate_ratios_by_sample[sample_idx].append((layer_idx, ratio))

            if layer_idx == self.after_layer_idx:
                after_layer_hidden_states = hidden_states.clone()

        hidden_states = language_model_self.norm(hidden_states)
        if after_layer_hidden_states is None:
            raise ValueError(f"Failed to capture hidden states for after_layer index {self.after_layer_idx}.")
        return (
            candidate_ratios_by_sample,
            after_layer_hidden_states,
            hidden_states,
            probe_cache,
        )

    def _run_after_layer_window_decode(
        self,
        language_model_self,
        input_attention_mask,
        text_position_ids: torch.Tensor,
        rotary_position_ids: torch.Tensor,
        probe_cache,
        probe_sample_idx: int,
        after_layer_hidden_states: torch.Tensor,
        selected_layer_indices: list[int],
        cache_position: torch.Tensor,
        visual_pos_masks,
        deepstack_visual_embeds,
        kwargs: dict,
    ) -> torch.Tensor:
        hidden_states = after_layer_hidden_states.clone()
        position_embeddings = language_model_self.rotary_emb(hidden_states, rotary_position_ids)
        attention_mask = None
        attention_mask_history_len = None

        for layer_idx in selected_layer_indices:
            decoder_layer = language_model_self.layers[layer_idx]
            layer_history_only_cache = self._build_history_only_cache(
                probe_cache,
                layer_idx=layer_idx,
                sample_idx=probe_sample_idx,
                config=language_model_self.config,
            )
            layer_history_len = layer_history_only_cache.get_seq_length()
            if attention_mask is None or attention_mask_history_len != layer_history_len:
                attention_mask = create_causal_mask(
                    config=language_model_self.config,
                    inputs_embeds=hidden_states,
                    attention_mask=input_attention_mask,
                    cache_position=cache_position,
                    past_key_values=layer_history_only_cache,
                    position_ids=text_position_ids,
                )
                attention_mask_history_len = layer_history_len
            original_layer_idx = decoder_layer.self_attn.layer_idx
            decoder_layer.self_attn.layer_idx = 0
            try:
                hidden_states, _ = self._forward_layer_with_attention(
                    decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    text_position_ids=text_position_ids,
                    past_key_values=layer_history_only_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            finally:
                decoder_layer.self_attn.layer_idx = original_layer_idx

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = language_model_self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        attention_mask = None
        attention_mask_history_len = None

        for layer_idx in range(self.after_layer_idx + 1, len(language_model_self.layers)):
            decoder_layer = language_model_self.layers[layer_idx]
            layer_history_only_cache = self._build_history_only_cache(
                probe_cache,
                layer_idx=layer_idx,
                sample_idx=probe_sample_idx,
                config=language_model_self.config,
            )
            layer_history_len = layer_history_only_cache.get_seq_length()
            if attention_mask is None or attention_mask_history_len != layer_history_len:
                attention_mask = create_causal_mask(
                    config=language_model_self.config,
                    inputs_embeds=hidden_states,
                    attention_mask=input_attention_mask,
                    cache_position=cache_position,
                    past_key_values=layer_history_only_cache,
                    position_ids=text_position_ids,
                )
                attention_mask_history_len = layer_history_len
            original_layer_idx = decoder_layer.self_attn.layer_idx
            decoder_layer.self_attn.layer_idx = 0
            try:
                hidden_states, _ = self._forward_layer_with_attention(
                    decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    text_position_ids=text_position_ids,
                    past_key_values=layer_history_only_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            finally:
                decoder_layer.self_attn.layer_idx = original_layer_idx

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = language_model_self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = language_model_self.norm(hidden_states)
        return hidden_states

    def _run_reforward_decode(
        self,
        language_model_self,
        inputs_embeds: torch.Tensor,
        input_attention_mask,
        text_position_ids: torch.Tensor,
        rotary_position_ids: torch.Tensor,
        past_key_values,
        cache_position: torch.Tensor,
        visual_pos_masks,
        deepstack_visual_embeds,
        kwargs: dict,
    ):
        (
            candidate_ratios_by_sample,
            after_layer_hidden_states,
            probe_final_hidden_states,
            probe_cache,
        ) = self._run_probe_decode(
            language_model_self=language_model_self,
            inputs_embeds=inputs_embeds,
            input_attention_mask=input_attention_mask,
            text_position_ids=text_position_ids,
            rotary_position_ids=rotary_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            kwargs=kwargs,
        )

        batch_size = inputs_embeds.shape[0]
        selected_layer_indices_by_sample = []
        selected_window_means = []
        for sample_idx, candidate_ratios in enumerate(candidate_ratios_by_sample):
            if not self.visual_token_indices_by_sample[sample_idx]:
                selected_layer_indices_by_sample.append([])
                selected_window_means.append(0.0)
                continue

            selected_layer_indices = self._select_top_k_layers(candidate_ratios)
            selected_layer_indices_by_sample.append(selected_layer_indices)
            selected_window_means.append(self._compute_window_mean(candidate_ratios, selected_layer_indices))

        if batch_size == 1:
            self.last_selected_layers = [layer_idx + 1 for layer_idx in selected_layer_indices_by_sample[0]]
            self.last_selected_window_means = [] if not self.last_selected_layers else selected_window_means[:1]
        else:
            self.last_selected_layers = [
                [selected_layer_idx + 1 for selected_layer_idx in selected_layer_indices]
                for selected_layer_indices in selected_layer_indices_by_sample
            ]
            self.last_selected_window_means = selected_window_means

        self._record_selected_windows(
            selected_layer_indices_by_sample=selected_layer_indices_by_sample,
            selected_window_means=selected_window_means,
            cache_position=cache_position,
        )

        updated_hidden_states = []
        for sample_idx, selected_layer_indices in enumerate(selected_layer_indices_by_sample):
            if not selected_layer_indices:
                updated_hidden_states.append(probe_final_hidden_states[sample_idx : sample_idx + 1])
                continue

            sample_deepstack_visual_embeds = self._slice_sample_deepstack_visual_embeds(
                deepstack_visual_embeds,
                visual_pos_masks,
                sample_idx,
            )
            sample_attention_mask = self._slice_sample_attention_mask(input_attention_mask, sample_idx)
            sample_text_position_ids = text_position_ids[sample_idx : sample_idx + 1]
            sample_rotary_position_ids = rotary_position_ids[:, sample_idx : sample_idx + 1, :]
            sample_visual_pos_masks = self._slice_sample_tensor(visual_pos_masks, sample_idx)
            sample_hidden_states = self._run_after_layer_window_decode(
                language_model_self=language_model_self,
                input_attention_mask=sample_attention_mask,
                text_position_ids=sample_text_position_ids,
                rotary_position_ids=sample_rotary_position_ids,
                probe_cache=probe_cache,
                probe_sample_idx=sample_idx,
                after_layer_hidden_states=after_layer_hidden_states[sample_idx : sample_idx + 1],
                selected_layer_indices=selected_layer_indices,
                cache_position=cache_position,
                visual_pos_masks=sample_visual_pos_masks,
                deepstack_visual_embeds=sample_deepstack_visual_embeds,
                kwargs=kwargs,
            )
            updated_hidden_states.append(sample_hidden_states)

        hidden_states = torch.cat(updated_hidden_states, dim=0)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=probe_cache,
        )

    def apply(self):
        def patched_forward(
            language_model_self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **kwargs,
        ):
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if use_cache and past_key_values is None and not torch.jit.is_tracing():
                past_key_values = DynamicCache(config=language_model_self.config)

            if inputs_embeds is None:
                inputs_embeds = language_model_self.embed_tokens(input_ids)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

            if position_ids is None:
                position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

            is_decode_step = self._is_decode_step(inputs_embeds, past_key_values)
            self._capture_visual_token_indices(
                visual_pos_masks,
                batch_size=inputs_embeds.shape[0],
                is_decode_step=is_decode_step,
            )
            input_attention_mask = attention_mask

            text_position_ids, rotary_position_ids = self._split_position_ids(position_ids)

            if self._should_split_prefill_for_first_token(inputs_embeds, past_key_values, use_cache):
                self._reset_selected_window_history(inputs_embeds.shape[0])
                if not any(self.visual_token_indices_by_sample):
                    self.last_selected_layers = []
                    self.last_selected_window_means = []
                    return self.original_forward(
                        input_ids=None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        visual_pos_masks=visual_pos_masks,
                        deepstack_visual_embeds=deepstack_visual_embeds,
                        **kwargs,
                    )

                prefix_seq_len = inputs_embeds.shape[1] - 1
                prefix_outputs = self.original_forward(
                    input_ids=None,
                    attention_mask=self._slice_sequence_tensor(attention_mask, prefix_seq_len),
                    position_ids=self._slice_sequence_tensor(position_ids, prefix_seq_len),
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds[:, :-1, :],
                    use_cache=use_cache,
                    cache_position=cache_position[:-1],
                    visual_pos_masks=self._slice_sequence_tensor(visual_pos_masks, prefix_seq_len),
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    **kwargs,
                )

                last_outputs = self._run_reforward_decode(
                    language_model_self=language_model_self,
                    inputs_embeds=inputs_embeds[:, -1:, :],
                    input_attention_mask=input_attention_mask,
                    text_position_ids=text_position_ids[:, -1:],
                    rotary_position_ids=rotary_position_ids[..., -1:],
                    past_key_values=past_key_values,
                    cache_position=cache_position[-1:],
                    visual_pos_masks=None,
                    deepstack_visual_embeds=None,
                    kwargs=kwargs,
                )
                hidden_states = torch.cat([prefix_outputs.last_hidden_state, last_outputs.last_hidden_state], dim=1)
                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=last_outputs.past_key_values,
                )

            if not is_decode_step:
                self._reset_selected_window_history(inputs_embeds.shape[0])
                self.last_selected_layers = []
                self.last_selected_window_means = []
                return self.original_forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    **kwargs,
                )

            if not any(self.visual_token_indices_by_sample):
                self.last_selected_layers = []
                self.last_selected_window_means = []
                return self.original_forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    **kwargs,
                )

            return self._run_reforward_decode(
                language_model_self=language_model_self,
                inputs_embeds=inputs_embeds,
                input_attention_mask=input_attention_mask,
                text_position_ids=text_position_ids,
                rotary_position_ids=rotary_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                kwargs=kwargs,
            )

        self.language_model.forward = MethodType(patched_forward, self.language_model)


@register_model("qwen3_vl_reforward_visual_attention_after_layer")
class Qwen3_VL_ReForwardVisualAttentionAfterLayer(Qwen3_VL_Base):
    @property
    def selected_window_history_by_sample(self):
        return self._reforward_visual_attention_patcher.selected_window_history_by_sample

    @property
    def last_selected_layers(self):
        return self._reforward_visual_attention_patcher.last_selected_layers

    @property
    def last_selected_window_means(self):
        return self._reforward_visual_attention_patcher.last_selected_window_means

    def _resolve_window_history_save_path(self) -> Path | None:
        if not self.save_window_history_path:
            return None

        path = Path(self.save_window_history_path)
        if path.exists() and path.is_dir():
            path = path / "window_history.json"

        if self.world_size > 1:
            if path.suffix:
                path = path.with_name(f"{path.stem}.rank{self.rank}{path.suffix}")
            else:
                path = path.with_name(f"{path.name}.rank{self.rank}")

        return path

    def _dump_window_history(self, window_history_records: list[dict]):
        save_path = self._resolve_window_history_save_path()
        if save_path is None:
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(window_history_records, f, ensure_ascii=False, indent=2)

    def _build_window_history_records(self, chunk_requests: list[Instance]) -> list[dict]:
        history_by_sample = self.selected_window_history_by_sample
        if len(history_by_sample) != len(chunk_requests):
            raise ValueError(
                "selected_window_history_by_sample size does not match chunk size: "
                f"{len(history_by_sample)} vs {len(chunk_requests)}."
            )

        records = []
        for request, sample_history in zip(chunk_requests, history_by_sample):
            records.append(
                {
                    "request_idx": int(request.idx),
                    "task": str(request.task_name),
                    "doc_id": str(request.doc_id),
                    "split": request.metadata.get("split"),
                    "window_history": [
                        {
                            "step": int(step_record["step"]),
                            "cache_position": int(step_record["cache_position"]),
                            "window": list(step_record["window"]),
                            "layers": list(step_record["layers"]),
                            "window_mean": float(step_record["window_mean"]),
                        }
                        for step_record in sample_history
                    ],
                }
            )
        return records

    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = []
        window_history_records = []

        def _collate(req: Instance):
            toks = self.tokenizer.encode(req.args[0])
            return -len(toks), req.args[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator(requests, _collate, group_fn=lambda req: req.args[1], grouping=True)
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._preprocess_chunk, [req.args for req in chunks[0]]) if chunks else None

            for idx in range(len(chunks)):
                chunk_requests = chunks[idx]
                inputs, contexts, gen_kwargs, until = future.result()

                if idx + 1 < len(chunks):
                    future = executor.submit(self._preprocess_chunk, [req.args for req in chunks[idx + 1]])

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                cont = self.model.generate(**inputs, **generate_kwargs)

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                window_history_records.extend(self._build_window_history_records(chunk_requests))

                for ans, context in zip(answers, contexts):
                    ans = self._strip_thinking(ans)
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

        res = re_ords.get_original(res)
        window_history_records = re_ords.get_original(window_history_records)
        self._dump_window_history(window_history_records)
        pbar.close()
        return res

    def __init__(
        self,
        after_layer: int = 35,
        candidate_start: int = 8,
        candidate_end: int = 35,
        top_k: int = 9,
        window_size: int | None = None,
        attn_implementation: str = "eager",
        **kwargs,
    ) -> None:
        if kwargs.get("device_map", "auto") == "auto" and kwargs.get("device") not in (None, "cuda"):
            kwargs["device_map"] = kwargs["device"]
        if kwargs.get("attn_implementation", attn_implementation) != "eager":
            raise ValueError("qwen3_vl_reforward_visual_attention_after_layer requires attn_implementation=eager")
        kwargs["attn_implementation"] = "eager"
        self.save_window_history_path = kwargs.pop("save_window_history_path", None)

        pretrained = kwargs.get("pretrained", "Qwen/Qwen3-VL-4B-Instruct")
        model_type = getattr(AutoConfig.from_pretrained(pretrained, trust_remote_code=True), "model_type", "")
        if not re.match(r"qwen3_vl", model_type):
            raise ValueError(
                "qwen3_vl_reforward_visual_attention_after_layer currently only supports Qwen3-VL checkpoints, "
                f"but got model_type='{model_type}' from '{pretrained}'."
            )

        super().__init__(**kwargs)

        self.after_layer = int(after_layer)
        self.candidate_start = int(candidate_start)
        self.candidate_end = int(candidate_end)
        self.window_size = int(top_k if window_size is None else window_size)
        self.top_k = self.window_size

        total_layers = len(self.model.model.language_model.layers)
        if self.after_layer < 1 or self.after_layer > total_layers:
            raise ValueError(f"after_layer must satisfy 1 <= after_layer <= {total_layers}, got {self.after_layer}.")
        if self.candidate_start < 1 or self.candidate_end > self.after_layer or self.candidate_start > self.candidate_end:
            raise ValueError(
                f"candidate layers must satisfy 1 <= start <= end <= after_layer={self.after_layer}, "
                f"got [{self.candidate_start}, {self.candidate_end}]."
            )
        if self.window_size < 1:
            raise ValueError(f"top_k/window_size must be positive, got {self.window_size}.")
        candidate_span = self.candidate_end - self.candidate_start + 1
        if self.window_size > candidate_span:
            raise ValueError(
                f"top_k/window_size must be <= candidate span {candidate_span}, got {self.window_size}."
            )

        patcher = VisualAttentionTopKReForwardHistoryOnlyPatcher(
            model=self.model,
            after_layer=self.after_layer,
            candidate_start=self.candidate_start,
            candidate_end=self.candidate_end,
            top_k=self.window_size,
        )
        patcher.apply()
        self._reforward_visual_attention_patcher = patcher
