"""Extended Qwen2.5 Language Model for SNAC token prediction.

Takes the pre-trained Qwen2.5 backbone and extends it with:
- Expanded vocabulary for SNAC audio tokens + emotion tags
- Speaker cross-attention layers injected at regular intervals
- Separate output heads for efficient SNAC level prediction
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

from leander_tts.model.attention import SpeakerCrossAttention


class LeanderLM(nn.Module):
    """Extended LLM for text-to-SNAC-token generation with speaker conditioning."""

    def __init__(
        self,
        backbone: str = "Qwen/Qwen2.5-1.5B",
        n_new_tokens: int = 12317,
        d_model: int = 1536,
        n_layers: int = 28,
        inject_every_n_layers: int = 4,
        n_heads: int = 12,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.d_model = d_model
        self.n_layers = n_layers
        self.inject_every_n_layers = inject_every_n_layers
        self.device = device
        self.dtype = dtype
        self._n_new_tokens = n_new_tokens

        # Will be initialized in load_backbone()
        self.backbone: Optional[nn.Module] = None
        self.speaker_cross_attn_layers: Optional[nn.ModuleDict] = None
        self.lm_head: Optional[nn.Linear] = None

    def load_backbone(self, pretrained: bool = True) -> None:
        """Load the Qwen2.5 backbone and set up extensions.

        Args:
            pretrained: If True, load pre-trained weights. If False, initialize randomly.
        """
        if pretrained:
            config = AutoConfig.from_pretrained(
                self.backbone_name, trust_remote_code=True,
            )
            self.backbone = AutoModelForCausalLM.from_pretrained(
                self.backbone_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                self.backbone_name, trust_remote_code=True,
            )
            self.backbone = AutoModelForCausalLM.from_config(
                config, torch_dtype=self.dtype,
            )

        # Expand embedding and LM head for new tokens
        self._expand_vocabulary()

        # Add speaker cross-attention layers
        self._add_speaker_cross_attention()

    def init_from_config(self, config_dict: dict) -> None:
        """Initialize model structure without loading pre-trained weights.

        Useful for testing with small random models.
        """
        config = AutoConfig.from_pretrained(
            self.backbone_name, trust_remote_code=True,
        )
        # Override config for testing
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.backbone = AutoModelForCausalLM.from_config(
            config, torch_dtype=self.dtype,
        )

        self.d_model = config.hidden_size
        self.n_layers = config.num_hidden_layers

        self._expand_vocabulary()
        self._add_speaker_cross_attention()

    def _expand_vocabulary(self) -> None:
        """Expand the embedding matrix and LM head for SNAC + emotion tokens."""
        model = self.backbone

        # Get the current embedding layer
        embed = model.get_input_embeddings()
        old_vocab_size = embed.weight.shape[0]
        new_vocab_size = old_vocab_size + self._n_new_tokens
        embed_dim = embed.weight.shape[1]

        # Create new larger embedding
        new_embed = nn.Embedding(new_vocab_size, embed_dim, dtype=embed.weight.dtype)
        with torch.no_grad():
            new_embed.weight[:old_vocab_size] = embed.weight
            # Initialize new tokens with small random values
            nn.init.normal_(new_embed.weight[old_vocab_size:], mean=0.0, std=0.02)

        model.set_input_embeddings(new_embed)

        # Expand LM head
        old_head = model.get_output_embeddings()
        if old_head is not None:
            old_out_features = old_head.weight.shape[0]
            new_head = nn.Linear(
                embed_dim, new_vocab_size, bias=old_head.bias is not None,
                dtype=old_head.weight.dtype,
            )
            with torch.no_grad():
                new_head.weight[:old_out_features] = old_head.weight
                nn.init.normal_(new_head.weight[old_out_features:], mean=0.0, std=0.02)
                if old_head.bias is not None:
                    new_head.bias[:old_out_features] = old_head.bias
                    new_head.bias[old_out_features:] = 0.0
            model.set_output_embeddings(new_head)

        self._total_vocab_size = new_vocab_size
        self._original_vocab_size = old_vocab_size

    def _add_speaker_cross_attention(self) -> None:
        """Add speaker cross-attention layers at regular intervals."""
        self.speaker_cross_attn_layers = nn.ModuleDict()

        # Inject at every N-th layer
        for layer_idx in range(
            self.inject_every_n_layers - 1,
            self.n_layers,
            self.inject_every_n_layers,
        ):
            self.speaker_cross_attn_layers[str(layer_idx)] = SpeakerCrossAttention(
                d_model=self.d_model,
                n_heads=8,
            ).to(self.dtype)

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on LLM layers for speaker cross-attention injection."""
        self._hooks = []
        self._current_speaker_embedding: Optional[torch.Tensor] = None

        # Access Qwen2 decoder layers
        if hasattr(self.backbone, "model"):
            layers = self.backbone.model.layers
        elif hasattr(self.backbone, "transformer"):
            layers = self.backbone.transformer.h
        else:
            raise ValueError(f"Unknown backbone structure: {type(self.backbone)}")

        for layer_idx_str, cross_attn in self.speaker_cross_attn_layers.items():
            layer_idx = int(layer_idx_str)
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(
                    self._make_speaker_hook(cross_attn)
                )
                self._hooks.append(hook)

    def _make_speaker_hook(self, cross_attn: SpeakerCrossAttention):
        """Create a forward hook that applies speaker cross-attention."""
        def hook(module, input, output):
            if self._current_speaker_embedding is None:
                return output

            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
                hidden_states = cross_attn(hidden_states, self._current_speaker_embedding)
                return (hidden_states,) + output[1:]
            else:
                return cross_attn(output, self._current_speaker_embedding)

        return hook

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [B, T] token IDs
            attention_mask: [B, T] attention mask
            speaker_embedding: [B, D] global speaker embedding (or None)
            labels: [B, T] target token IDs for loss computation
            past_key_values: KV cache for generation
            use_cache: whether to return KV cache

        Returns:
            Dict with 'logits', optionally 'loss', and optionally 'past_key_values'
        """
        # Set speaker embedding for hooks
        self._current_speaker_embedding = speaker_embedding

        # Forward through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Clear speaker embedding
        self._current_speaker_embedding = None

        result = {"logits": outputs.logits}

        if labels is not None:
            result["loss"] = outputs.loss

        if use_cache:
            result["past_key_values"] = outputs.past_key_values

        return result

    @torch.inference_mode()
    def generate_streaming(
        self,
        input_ids: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        audio_end_token_id: int = 0,
    ):
        """Generate SNAC tokens autoregressively with streaming.

        Yields one token at a time for streaming to the detokenizer.

        Args:
            input_ids: [1, T] text prompt token IDs
            speaker_embedding: [1, D] speaker embedding
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            repetition_penalty: penalize repeated tokens
            audio_end_token_id: token ID for AUDIO_END (stop signal)

        Yields:
            int: next token ID
        """
        self._current_speaker_embedding = speaker_embedding

        past_key_values = None
        current_ids = input_ids
        generated_ids = []

        for _ in range(max_new_tokens):
            outputs = self.backbone(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            # Repetition penalty
            if repetition_penalty != 1.0 and generated_ids:
                for prev_id in set(generated_ids[-100:]):
                    if logits[0, prev_id] > 0:
                        logits[0, prev_id] /= repetition_penalty
                    else:
                        logits[0, prev_id] *= repetition_penalty

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove,
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            token_id = next_token.item()

            generated_ids.append(token_id)
            current_ids = next_token

            yield token_id

            if token_id == audio_end_token_id:
                break

        self._current_speaker_embedding = None

    def freeze_backbone(self) -> None:
        """Freeze original backbone parameters (for stage 3a training)."""
        if self.backbone is None:
            return
        for name, param in self.backbone.named_parameters():
            # Don't freeze new embedding rows
            param.requires_grad = False

        # Unfreeze new token embeddings
        embed = self.backbone.get_input_embeddings()
        embed.weight.requires_grad = True

        head = self.backbone.get_output_embeddings()
        if head is not None:
            head.weight.requires_grad = True
            if head.bias is not None:
                head.bias.requires_grad = True

        # Speaker cross-attention is always trainable
        if self.speaker_cross_attn_layers is not None:
            for param in self.speaker_cross_attn_layers.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters (for stage 3b training)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
