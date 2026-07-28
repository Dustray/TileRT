"""验证 QwenTransformerStack 的 golden 路径。

本脚本加载真实 HF 权重（Qwen3.6-35B-A3B），将 prompt token 经过
embed -> 40 层异构 TransformerStack -> final RMSNorm -> lm_head，
用 greedy decode 输出可读的生成文本，以验证 golden_forward 端到端可用。

执行：
    cd /public/home/dinggy/yiny/projects/TileRT
    PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT python3 scripts/verify_qwen36_transformer_stack_forward.py
"""
import json
import logging
import os
import sys
import time
import warnings

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from tilert import logger

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)

MODEL_DIR = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
PROMPT = "How many r's are in the word strawberry? Think step by step."
MAX_NEW_TOKENS = 20
DEVICE_ID = 0
NUM_DEVICES = 8


def load_layer_state_dict(model_dir: str, layer_idx: int) -> dict[str, torch.Tensor]:
    """仅加载指定层所需的分片，去掉 `model.language_model.layers.N.` 前缀。"""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    # 收集该层所有键所在的文件。
    files_for_layer = sorted({wm[k] for k in wm if k.startswith(prefix)})
    out: dict[str, torch.Tensor] = {}
    for fn in files_for_layer:
        path = os.path.join(model_dir, fn)
        logger.info("Layer %d: loading %s...", layer_idx, fn)
        partial = load_file(path, device="cpu")
        for k, v in partial.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v.to(f"cuda:{DEVICE_ID}")
    return out


def load_text_head_weights(model_dir: str) -> dict[str, torch.Tensor]:
    """加载 embed_tokens / final norm / lm_head 三个头部权重。"""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    head_keys = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    files = sorted({wm[k] for k in head_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        partial = load_file(os.path.join(model_dir, fn), device="cpu")
        for k in head_keys:
            if k in partial:
                sd[k] = partial[k].to(f"cuda:{DEVICE_ID}")
    return sd


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3.6 的 RMSNorm: x / rms * (1 + gamma)。"""
    x_f = x.float()
    norm = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return norm * (1.0 + weight.float())


def main():
    torch.set_num_threads(64)
    torch.manual_seed(42)
    # Allow fragmentation recovery when the two GPUs are nearly full.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    logger.info("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_tokens = tokenizer.encode(chat_input, add_special_tokens=False)
    logger.info("Prompt tokens: %d -> %s...", len(prompt_tokens), prompt_tokens[:10])

    logger.info("[2/5] Building QwenTransformerStack on cuda:%s...", DEVICE_ID)
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.transformer_stack import QwenTransformerStack
    from tilert.models.qwen3_6.modules.moe import QwenMoeBlock
    from tilert.models.utils import precompute_mrope_embed

    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    shared_ffn = QwenMoeBlock(
        model_args=model_args, device_id=DEVICE_ID, num_devices=NUM_DEVICES
    )
    stack = QwenTransformerStack(
        model_args=model_args,
        device_id=DEVICE_ID,
        num_devices=NUM_DEVICES,
        cached_ffn_ops=[shared_ffn] * model_args.n_layers,
    )

    logger.info("[3/5] Loading real checkpoint weights layer by layer...")
    for layer_idx, block in enumerate(stack.exec_seq):
        lsd = load_layer_state_dict(MODEL_DIR, layer_idx)
        block.init_reference_weights(lsd)
        del lsd
        torch.cuda.empty_cache()

    head_sd = load_text_head_weights(MODEL_DIR)
    embed_tokens = head_sd["model.language_model.embed_tokens.weight"].to(
        f"cuda:{DEVICE_ID}", torch.bfloat16
    )
    final_norm_w = head_sd["model.language_model.norm.weight"].to(f"cuda:{DEVICE_ID}")
    lm_head_w = head_sd["lm_head.weight"].to(f"cuda:{DEVICE_ID}", torch.bfloat16)
    del head_sd
    logger.info("Weights loaded; final norm/lm_head ready.")

    logger.info("[4/5] Running greedy generation with golden_forward...")
    # precompute_mrope_embed 已经返回 (cos, sin) tuple，无需 view_as_real
    cos_embed, sin_embed = precompute_mrope_embed(model_args)
    mrope_embed = (
        cos_embed.to(f"cuda:{DEVICE_ID}").float(),
        sin_embed.to(f"cuda:{DEVICE_ID}").float(),
    )

    caches = None
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    cur_tok = torch.tensor([[prompt_tokens[0]]], dtype=torch.long, device=f"cuda:{DEVICE_ID}")

    t0 = time.time()
    n_steps = len(prompt_tokens) + MAX_NEW_TOKENS
    for pos in range(n_steps):
        if pos == 0:
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=f"cuda:{DEVICE_ID}")
            x = embed_tokens[input_ids].to(torch.bfloat16)
            start_pos = 0
        else:
            input_ids = cur_tok
            x = embed_tokens[input_ids].to(torch.bfloat16)
            start_pos = pos - 1

        hidden, caches = stack.golden_forward(
            x, start_pos=start_pos, caches=caches, mrope_embed=mrope_embed
        )
        hidden_norm = rmsnorm(hidden[:, -1:, :], final_norm_w)
        # Chunked matmul for the large lm_head to avoid a single big allocation
        # when GPU memory is already nearly exhausted.
        vocab_size = lm_head_w.shape[0]
        chunk_size = 32768
        logits_chunks = []
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            logits_chunks.append(
                (hidden_norm.float() @ lm_head_w[start:end].T.float())
            )
            torch.cuda.synchronize()
        logits = torch.cat(logits_chunks, dim=-1).squeeze(1)
        next_id = int(logits.argmax(-1).item())

        if pos >= len(prompt_tokens):
            generated.append(next_id)
            if next_id == eos_id:
                break
            cur_tok = torch.tensor([[next_id]], dtype=torch.long, device=f"cuda:{DEVICE_ID}")
        else:
            if pos + 1 < len(prompt_tokens):
                cur_tok = torch.tensor(
                    [[prompt_tokens[pos + 1]]], dtype=torch.long, device=f"cuda:{DEVICE_ID}"
                )
            else:
                cur_tok = torch.tensor([[next_id]], dtype=torch.long, device=f"cuda:{DEVICE_ID}")

    dt = time.time() - t0
    ms_per_tok = 1000 * dt / max(1, len(generated))
    logger.info(
        "Generated %d tokens in %.2fs (%.1f ms/tok)", len(generated), dt, ms_per_tok
    )

    text = tokenizer.decode(generated, skip_special_tokens=False)
    print("\n===== GENERATED TEXT =====\n", text)
    print("===== TOKEN IDS =====\n", generated)

    logger.info("\n=== Transformer stack golden forward text-generation test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logger.exception("\n=== FAILED: %s ===", exc)
        raise
