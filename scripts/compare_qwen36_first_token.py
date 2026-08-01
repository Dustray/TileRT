"""Compare TileRT 8-GPU golden first-token logits against the HF reference.

HF reference for the prompt below (from a previous transformers 5.5.0 run):
  first token id: 8160 ("Here")
  top10 ids:  [8160, 90700, 97237, 760, 1596, 31248, 77264, 1421, 107680, 97490]
  top10 vals: [24.75, 20.125, 18.5, 18.0, 17.5, 16.75, 16.5, 15.125, 15.0, 14.875]
"""
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

# Recorded HF first-token logits (transformers 5.5.0, bf16, sdpa).
HF_FIRST_TOKEN_ID = 8160
HF_TOP10_IDS = [8160, 90700, 97237, 760, 1596, 31248, 77264, 1421, 107680, 97490]
HF_TOP10_VALS = [24.75, 20.125, 18.5, 18.0, 17.5, 16.75, 16.5, 15.125, 15.0, 14.875]


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

    # Feed the prompt token-by-token; use no_grad and empty cache to control memory.
    cur_pos = 0
    with torch.no_grad():
        for token_id in prompt_tokens:
            tid = torch.tensor(token_id, dtype=torch.int32)
            layer.forward(tid, with_mtp=False, cur_pos=cur_pos)
            cur_pos += 1
            if cur_pos % 4 == 0:
                torch.cuda.empty_cache()

    # Collect full logits from every device to check for per-device divergence.
    per_device_logits = []
    for did in range(layer.num_devices):
        intermediates = layer._get_device_result(did)[0]
        logits = intermediates[layer.Idx.LOGITS_OUT][0, 0, :].float().cpu()
        per_device_logits.append(logits)

    # Compare devices 1..7 against device 0.
    ref_dev0 = per_device_logits[0]
    print("\n===== PER-DEVICE LOGITS DIVERGENCE =====")
    all_match = True
    for did in range(1, layer.num_devices):
        diff = (per_device_logits[did] - ref_dev0).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        argmax_match = per_device_logits[did].argmax().item() == ref_dev0.argmax().item()
        print(
            f"dev {did} vs dev 0: max_abs_diff={max_diff:.4f}  mean_abs_diff={mean_diff:.4f}  "
            f"argmax_match={argmax_match}"
        )
        if max_diff > 1e-3:
            all_match = False

    # Compare TileRT device 0 logits against the recorded HF reference.
    tilert_top = ref_dev0.topk(10)
    print("\n===== TileRT dev0 top-10 =====")
    print("ids :", tilert_top.indices.tolist())
    print("vals:", [round(v, 3) for v in tilert_top.values.tolist()])

    print("\n===== HF reference top-10 =====")
    print("ids :", HF_TOP10_IDS)
    print("vals:", HF_TOP10_VALS)

    # Numerical diff against HF at the top-10 positions.
    hf_topk_tensor = torch.tensor(HF_TOP10_VALS, dtype=torch.float32)
    tilert_at_hf_top = ref_dev0[torch.tensor(HF_TOP10_IDS)].float()
    print("\n===== HF vs TileRT at HF top-10 positions =====")
    print("HF vals   :", HF_TOP10_VALS)
    print("TileRT vals:", [round(v, 3) for v in tilert_at_hf_top.tolist()])
    print(
        "max_abs_diff(top10)=",
        (tilert_at_hf_top - hf_topk_tensor).abs().max().item(),
    )

    tilert_argmax = int(ref_dev0.argmax().item())
    print(f"\nTileRT argmax token id={tilert_argmax} ({tokenizer.decode([tilert_argmax])})")
    print(f"HF argmax token id={HF_FIRST_TOKEN_ID} ({tokenizer.decode([HF_FIRST_TOKEN_ID])})")

    layer.cleanup()
    return 0 if all_match and tilert_argmax == HF_FIRST_TOKEN_ID else 1


if __name__ == "__main__":
    sys.exit(main())
