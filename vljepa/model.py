"""VL-JEPA for temporal CXR (option 2).

    prior + current  ─►  BioViL-T image encoder (pair)  ─►  visual tokens
    query text       ─►  BioViL-T text encoder          ─►  query tokens
                         └─► Llama predictor (last 8 Llama-3.2-1B layers,
                             bidirectional) ─► predicted target embedding Ŝ
    class phrases    ─►  BioViL-T text encoder          ─►  S_Y^{1..5}
                         InfoNCE(Ŝ, S_Y) with the gold class as the positive
                         and the other four ``{Finding} is {class}.`` phrases
                         as in-example negatives.

The Llama stack matches VL-JEPA (Chen et al., arXiv:2512.10942): last 8
Llama-3.2-1B transformer layers, causal mask disabled, mean-pool non-pad
tokens, linear map into the BioViL-T 128-d text space. Query/target
tokenization stays on BioViL-T (not the Llama tokenizer).
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _root  # noqa: F401  — repo root on sys.path

from progression_phrases import CLS_ORDER
from tempcxr.modules.image_encoder_jepa import BioViLTImageEncoderJEPA
from tempcxr.modules.text_encoder import BioViLTTextEncoder

from .prompts import class_target_texts, query_text

LLAMA_NAME_DEFAULT = os.environ.get("VLJEPA_LLAMA_NAME", "meta-llama/Llama-3.2-1B")
N_LLAMA_LAYERS_DEFAULT = 8
N_CLS = len(CLS_ORDER)

# Llama-3.2-1B config (used when hub weights are unavailable).
_LLAMA32_1B = dict(
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=16,
    num_attention_heads=32,
    num_key_value_heads=8,
    rms_norm_eps=1e-5,
    vocab_size=128256,
    max_position_embeddings=2048,
    rope_theta=500000.0,
    attention_bias=False,
    mlp_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    initializer_range=0.02,
    use_cache=False,
)


def _llama_config(smoke: bool):
    from transformers import LlamaConfig

    if smoke:
        return LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=512,
            rope_theta=10000.0,
            attention_bias=False,
            mlp_bias=False,
            use_cache=False,
        )
    return LlamaConfig(**_LLAMA32_1B)


def _force_eager_bidirectional(module: nn.Module) -> None:
    """Prefer eager attention and turn off per-layer causal flags."""
    for m in module.modules():
        if hasattr(m, "is_causal"):
            try:
                m.is_causal = False
            except Exception:
                pass
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"


def _additive_pad_mask(pad_bool: torch.Tensor) -> torch.Tensor:
    """``pad_bool`` is True on valid tokens, shape ``(B, S)``.

    Returns a 4-D additive mask ``(B, 1, S, S)``: 0 where both query and
    key are valid, ``-inf`` if either is pad. No causal term — full
    bidirectional attention over the valid set.
    """
    valid = pad_bool.bool()
    attend = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, S, S)
    mask = torch.zeros(
        attend.shape, dtype=torch.float32, device=pad_bool.device
    )
    mask = mask.masked_fill(~attend, torch.finfo(mask.dtype).min)
    return mask.unsqueeze(1)


class LlamaPredictor(nn.Module):
    """Bidirectional Llama stack: visual tokens + query tokens → Ŝ."""

    def __init__(
        self,
        vis_dim: int = 128,
        txt_dim: int = 128,
        out_dim: int = 128,
        n_layers: int = N_LLAMA_LAYERS_DEFAULT,
        smoke: bool = False,
        llama_name: str = LLAMA_NAME_DEFAULT,
        llama_local: Optional[str] = None,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.smoke = bool(smoke)
        self.n_layers = int(n_layers)
        self.gradient_checkpointing = bool(gradient_checkpointing) and not smoke
        self.init_source = "random"
        hidden = self._build_llama_stack(
            n_layers=self.n_layers,
            smoke=self.smoke,
            llama_name=llama_name,
            llama_local=llama_local,
        )
        self.hidden_size = hidden
        self.vis_proj = nn.Linear(vis_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)
        self.type_embed = nn.Embedding(2, hidden)  # 0=vision, 1=query
        self.out_proj = nn.Linear(hidden, out_dim)
        self.out_norm = nn.LayerNorm(hidden)

    def _build_llama_stack(
        self,
        n_layers: int,
        smoke: bool,
        llama_name: str,
        llama_local: Optional[str],
    ) -> int:
        from transformers import LlamaModel

        rotary = None
        layers = None
        norm = None
        hidden = None

        if not smoke:
            load_src = llama_local or llama_name
            try:
                kwargs: Dict[str, Any] = {
                    "attn_implementation": "eager",
                    "torch_dtype": torch.float32,
                    "use_safetensors": True,
                }
                # local dir vs hub id
                if llama_local:
                    kwargs["local_files_only"] = True
                llama = LlamaModel.from_pretrained(load_src, **kwargs)
                all_layers = list(llama.layers)
                take = min(n_layers, len(all_layers))
                layers = nn.ModuleList(all_layers[-take:])
                norm = llama.norm
                rotary = getattr(llama, "rotary_emb", None)
                hidden = int(llama.config.hidden_size)
                self.n_layers = take
                self.init_source = f"pretrained:{load_src}:last{take}"
            except Exception as exc:
                print(
                    f"[vljepa] Llama weights not loaded from {load_src!r} "
                    f"({type(exc).__name__}: {exc}). "
                    f"Falling back to random-init Llama-3.2-1B last-{n_layers} "
                    f"architecture. Set HF_TOKEN / VLJEPA_LLAMA_LOCAL to use "
                    f"the paper init."
                )

        if layers is None:
            cfg = _llama_config(smoke)
            cfg._attn_implementation = "eager"
            cfg.num_hidden_layers = n_layers if not smoke else cfg.num_hidden_layers
            # Build a full model then keep the requested tail so layer
            # indices / RoPE match a real LlamaModel.
            full = LlamaModel(cfg)
            take = min(n_layers, len(full.layers))
            layers = nn.ModuleList(list(full.layers)[-take:])
            norm = full.norm
            rotary = getattr(full, "rotary_emb", None)
            hidden = int(cfg.hidden_size)
            self.n_layers = take
            if smoke:
                self.init_source = f"smoke:tiny:{take}L{hidden}d"
            elif self.init_source == "random":
                self.init_source = f"random:llama3.2-1b:{take}L"

        self.layers = layers
        self.norm = norm
        self.rotary_emb = rotary
        _force_eager_bidirectional(self)
        return int(hidden)

    def _position_embeddings(
        self, hidden: torch.Tensor, position_ids: torch.Tensor
    ):
        if self.rotary_emb is None:
            return None
        try:
            return self.rotary_emb(hidden, position_ids)
        except TypeError:
            return None

    def _run_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        attn_4d: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings,
    ) -> torch.Tensor:
        sig = inspect.signature(layer.forward)
        kwargs: Dict[str, Any] = {}
        if "attention_mask" in sig.parameters:
            kwargs["attention_mask"] = attn_4d
        if "position_ids" in sig.parameters:
            kwargs["position_ids"] = position_ids
        if (
            "position_embeddings" in sig.parameters
            and position_embeddings is not None
        ):
            kwargs["position_embeddings"] = position_embeddings
        if "output_attentions" in sig.parameters:
            kwargs["output_attentions"] = False
        if "use_cache" in sig.parameters:
            kwargs["use_cache"] = False
        if "cache_position" in sig.parameters:
            kwargs["cache_position"] = position_ids[0]
        out = layer(hidden, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    def forward(
        self,
        vis_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        txt_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        """
        vis_tokens : (B, Nv, Dv)
        txt_tokens : (B, Nt, Dt)
        txt_mask   : (B, Nt) bool, True = keep
        """
        bsz, n_vis, _ = vis_tokens.shape
        n_txt = txt_tokens.shape[1]
        vis_h = self.vis_proj(vis_tokens.float())
        txt_h = self.txt_proj(txt_tokens.float())
        vis_h = vis_h + self.type_embed.weight[0]
        txt_h = txt_h + self.type_embed.weight[1]
        hidden = torch.cat([vis_h, txt_h], dim=1)  # (B, Nv+Nt, H)
        seq_len = hidden.shape[1]

        vis_mask = torch.ones(
            bsz, n_vis, dtype=torch.bool, device=hidden.device
        )
        token_mask = torch.cat([vis_mask, txt_mask.bool()], dim=1)
        attn_4d = _additive_pad_mask(token_mask).to(dtype=hidden.dtype)
        position_ids = torch.arange(
            seq_len, device=hidden.device, dtype=torch.long
        ).unsqueeze(0).expand(bsz, -1)
        pos_emb = self._position_embeddings(hidden, position_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._run_layer,
                    layer,
                    hidden,
                    attn_4d,
                    position_ids,
                    pos_emb,
                    use_reentrant=False,
                )
            else:
                hidden = self._run_layer(
                    layer, hidden, attn_4d, position_ids, pos_emb
                )

        if self.norm is not None:
            hidden = self.norm(hidden)
        hidden = self.out_norm(hidden)

        mask_f = token_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        pred = F.normalize(self.out_proj(pooled), dim=-1)

        if not return_aux:
            return pred
        aux = {
            "vis_proj": tuple(vis_h.shape),
            "txt_proj": tuple(txt_h.shape),
            "predictor_tokens": tuple(hidden.shape),
            "token_mask": tuple(token_mask.shape),
            "attn_4d": tuple(attn_4d.shape),
            "pooled": tuple(pooled.shape),
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "init_source": self.init_source,
        }
        return pred, aux


class VLJEPA(nn.Module):
    """Pair image + finding query → progression-phrase embedding."""

    def __init__(
        self,
        image_mode: str = "biovilt",
        smoke: bool = False,
        n_llama_layers: int = N_LLAMA_LAYERS_DEFAULT,
        llama_name: str = LLAMA_NAME_DEFAULT,
        llama_local: Optional[str] = None,
        freeze_image_encoder: bool = True,
        freeze_text_encoder: bool = False,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        img_mode = "biovilt_no_pretrained" if smoke else image_mode
        self.image_encoder = BioViLTImageEncoderJEPA(mode=img_mode)
        self.text_encoder = BioViLTTextEncoder(mode="biovilt")
        vis_dim = int(self.image_encoder.embed_dim)
        txt_dim = int(self.text_encoder.proj_dim)
        self.predictor = LlamaPredictor(
            vis_dim=vis_dim,
            txt_dim=txt_dim,
            out_dim=txt_dim,
            n_layers=n_llama_layers,
            smoke=smoke,
            llama_name=llama_name,
            llama_local=llama_local,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.freeze_image_encoder = bool(freeze_image_encoder)
        self.freeze_text_encoder = bool(freeze_text_encoder)
        if self.freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            self.image_encoder.eval()
        if self.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_image_encoder:
            self.image_encoder.eval()
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def encode_images(
        self, prior: torch.Tensor, current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BioViL-T pair encoder. Returns (global, patches), both unit-norm."""
        ctx = torch.no_grad() if self.freeze_image_encoder else torch.enable_grad()
        with ctx:
            global_emb, patches = self.image_encoder(current, prior)
        return global_emb, patches

    def encode_text(self, texts: List[str]):
        ctx = torch.no_grad() if self.freeze_text_encoder else torch.enable_grad()
        with ctx:
            return self.text_encoder.forward_contrastive(texts)

    def predict_from_tokens(
        self,
        img_global: torch.Tensor,
        img_patches: torch.Tensor,
        query_local: torch.Tensor,
        query_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        vis = torch.cat([img_global.unsqueeze(1), img_patches], dim=1)
        return self.predictor(
            vis, query_local, query_mask, return_aux=return_aux
        )

    def forward(
        self,
        prior: torch.Tensor,
        current: torch.Tensor,
        query_texts: List[str],
        target_texts: Optional[List[str]] = None,
        return_aux: bool = False,
    ) -> Dict[str, Any]:
        img_g, img_p = self.encode_images(prior, current)
        q_g, q_loc, q_mask = self.encode_text(query_texts)
        pred = self.predict_from_tokens(
            img_g, img_p, q_loc, q_mask, return_aux=return_aux
        )
        aux = None
        if return_aux:
            pred, aux = pred
        out: Dict[str, Any] = {
            "pred": pred,
            "img_global": img_g,
            "img_patches": img_p,
            "query_global": q_g,
            "query_local": q_loc,
            "query_mask": q_mask,
        }
        if target_texts is not None:
            t_g, t_loc, t_mask = self.encode_text(target_texts)
            out["target_global"] = t_g
            out["target_local"] = t_loc
            out["target_mask"] = t_mask
        if aux is not None:
            out["aux"] = aux
        return out

    @torch.no_grad()
    def score_classes(
        self,
        prior: torch.Tensor,
        current: torch.Tensor,
        finding: str,
        text_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return ``(C,)`` cosine scores of Ŝ vs the 5 class phrases."""
        queries = [query_text(finding)]
        targets = class_target_texts(finding)
        cache = text_cache if text_cache is not None else {}

        def _cached(text: str) -> torch.Tensor:
            if text not in cache:
                g, _, _ = self.encode_text([text])
                cache[text] = g.detach()
            return cache[text]

        q_g = _cached(queries[0])
        # still need query *tokens* for the predictor
        _, q_loc, q_mask = self.encode_text(queries)
        img_g, img_p = self.encode_images(prior, current)
        pred = self.predict_from_tokens(img_g, img_p, q_loc, q_mask)
        tgt = torch.cat([_cached(t) for t in targets], dim=0)  # (C, D)
        pred_n = F.normalize(pred.float(), dim=-1)
        tgt_n = F.normalize(tgt.float(), dim=-1)
        return (pred_n * tgt_n).sum(dim=-1).squeeze(0)


def class_infonce_loss(
    pred: torch.Tensor,
    target_global: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In-example 5-way InfoNCE.

    ``pred``           (B, D) unit-norm predicted Ŝ
    ``target_global``  (B*C, D) or (B, C, D) class phrase embeddings
    ``labels``         (B,) gold class index
    """
    bsz = pred.shape[0]
    if target_global.dim() == 2:
        n_cls = target_global.shape[0] // bsz
        targets = target_global.view(bsz, n_cls, -1)
    else:
        targets = target_global
        n_cls = targets.shape[1]
    pred_n = F.normalize(pred.float(), dim=-1)
    tgt_n = F.normalize(targets.float(), dim=-1)
    logits = torch.einsum("bd,bcd->bc", pred_n, tgt_n) / float(temperature)
    loss = F.cross_entropy(logits, labels, weight=class_weights)
    return loss, logits
