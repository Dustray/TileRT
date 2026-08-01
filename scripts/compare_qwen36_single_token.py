"""Single-token TileRT forward to verify per-device convergence quickly."""
import logging
import sys
import warnings

import torch
from transformers import AutoTokenizer

from tilert import logger

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)

MODEL_DIR = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
PROMPT = "How many r's are in the word strawberry? Think step by step."


def build_prompt_tokens(tokenizer, prompt: str) -> list[int]:
    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(chat_input, add_special_tokens=False)


def main():
    torch.set_num_threads(64)
    torch.manual_seed(42)

    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.end2end import QwenShowHandsLayer
    from tilert.models.qwen3_6.temp_var_indices import Idx

    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    logger.info("Building QwenShowHandsLayer on %d devices...", torch.cuda.device_count())
    layer = QwenShowHandsLayer(
        model_args=model_args,
        model_path=MODEL_DIR,
        with_mtp=False,
        temperature=1.0,
        top_p=0.9,
        top_k=256,
        use_topp=False,
    )
    logger.info("Loading HF pretrained weights on %d devices...", layer.num_devices)
    layer.from_pretrained(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    prompt_tokens = build_prompt_tokens(tokenizer, PROMPT)
    logger.info("Prompt tokens: %d -> %s...", len(prompt_tokens), prompt_tokens[:10])

    # Run only the first token deterministically.
    first_tid = torch.tensor(prompt_tokens[0], dtype=torch.int32)
    with torch.no_grad():
        layer.forward(first_tid, with_mtp=False, cur_pos=0)

    per_device_logits = []
    for did in range(layer.num_devices):
        logits = layer.get_logits(did)[0, 0, :].float().cpu()
        per_device_logits.append(logits)

    ref_dev0 = per_device_logits[0]
    print("\n===== PER-DEVICE LOGITS DIVERGENCE (single token) =====")
    all_match = True
    for did in range(1, layer.num_devices):
        diff = (per_device_logits[did] - ref_dev0).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        argmax_match = per_device_logits[did].argmax().item() == ref_dev0.argmax().item()
        print(
            f"dev {did} vs dev 0: max_abs_diff={max_diff:.6f}  mean_abs_diff={mean_diff:.6f}  "
            f"argmax_match={argmax_match}"
        )
        if max_diff > 1e-3:
            all_match = False

    tilert_top = ref_dev0.topk(10)
    print("\n===== TileRT dev0 top-10 =====")
    print("ids :", tilert_top.indices.tolist())
    print("vals:", [round(v, 3) for v in tilert_top.values.tolist()])

    tilert_argmax = int(ref_dev0.argmax().item())
    print(f"\nTileRT argmax token id={tilert_argmax} ({tokenizer.decode([tilert_argmax])})")

    layer.cleanup()
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
