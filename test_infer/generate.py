"""
Generate text from a trained parameter-golf checkpoint.

Usage:
    python3 test_infer/generate.py                          # default prompt
    python3 test_infer/generate.py --prompt "The meaning of"
    python3 test_infer/generate.py --model final_model.pt --tokens 200
    python3 test_infer/generate.py --temperature 0.8 --top_k 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

from train_gpt import (
    GPT,
    CastedLinear,
    Hyperparameters,
    dequantize_state_dict_int8,
    restore_low_dim_params_to_fp32,
)


def get_logits(model: GPT, input_ids: Tensor) -> Tensor:
    """Forward pass that returns logits instead of loss."""
    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[Tensor] = []

    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, x0)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[model.num_encoder_layers + i](x, x0)

    x = model.final_norm(x)
    if model.tie_embeddings:
        logits_proj = F.linear(x, model.tok_emb.weight)
    else:
        logits_proj = model.lm_head(x)
    return model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)


@torch.no_grad()
def generate(
    model: GPT,
    input_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tensor:
    """Autoregressive generation."""
    seq = input_ids.clone()
    for _ in range(max_new_tokens):
        # Crop to last seq_len tokens if needed
        x = seq[:, -1024:]
        logits = get_logits(model, x)
        logits = logits[:, -1, :].float()
        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
    return seq


def load_model(model_path: str, device: torch.device) -> GPT:
    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )

    if model_path.endswith(".ptz"):
        import io, zlib
        with open(model_path, "rb") as f:
            blob = f.read()
        quant_state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
        state_dict = dequantize_state_dict_int8(quant_state)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=True)
    model.to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model.pt", help="Checkpoint path")
    parser.add_argument("--prompt", default="The", help="Prompt text")
    parser.add_argument("--tokens", type=int, default=150, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = load_model(args.model, device)

    prompt_ids = sp.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"Model: {args.model}")
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Tokens: {args.tokens}, temp={args.temperature}, top_k={args.top_k}")
    print("=" * 60)

    for i in range(args.num_samples):
        output = generate(model, input_ids, args.tokens, args.temperature, args.top_k)
        text = sp.decode(output[0].tolist())
        print(f"\n--- Sample {i+1} ---")
        print(text)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
