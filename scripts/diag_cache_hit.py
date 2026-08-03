"""Check whether full head projection cache hits across tokens."""
import logging
import warnings

import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s")

MODEL_DIR = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
PROMPT = "How many r's are in the word strawberry? Think step by step."


def main():
    torch.set_num_threads(64)
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.end2end import QwenShowHandsLayer

    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1
    layer = QwenShowHandsLayer(model_args=model_args, model_path=MODEL_DIR, with_mtp=False)
    layer.from_pretrained(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    chat = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.encode(chat, add_special_tokens=False)

    with torch.no_grad():
        for i, tid in enumerate(tokens[:4]):
            print(f"\n=== TOKEN {i} ===")
            layer.forward(torch.tensor(tid, dtype=torch.int32), with_mtp=False, cur_pos=i)
            print(f"dev0 allocated after token {i}: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
