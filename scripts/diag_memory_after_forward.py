"""Memory diagnostic after single forward."""
import logging
import warnings

import torch
from transformers import AutoTokenizer

from tilert import logger

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s")

MODEL_DIR = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
PROMPT = "How many r's are in the word strawberry? Think step by step."


def main():
    torch.set_num_threads(64)
    torch.manual_seed(42)
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.end2end import QwenShowHandsLayer

    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    layer = QwenShowHandsLayer(
        model_args=model_args,
        model_path=MODEL_DIR,
        with_mtp=False,
        temperature=1.0,
        top_p=0.9,
        top_k=256,
        use_topp=False,
    )
    layer.from_pretrained(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_tokens = tokenizer.encode(chat_input, add_special_tokens=False)
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens[:10]}...")

    # print memory after load
    for did in range(layer.num_devices):
        with torch.cuda.device(did):
            print(f"After load  dev{did}: allocated={torch.cuda.memory_allocated(did)/1e9:.2f} GB  reserved={torch.cuda.memory_reserved(did)/1e9:.2f} GB")

    with torch.no_grad():
        for i, tid in enumerate(prompt_tokens[:4]):
            layer.forward(torch.tensor(tid, dtype=torch.int32), with_mtp=False, cur_pos=i)
            for did in range(layer.num_devices):
                with torch.cuda.device(did):
                    print(f"After token {i} dev{did}: allocated={torch.cuda.memory_allocated(did)/1e9:.2f} GB  reserved={torch.cuda.memory_reserved(did)/1e9:.2f} GB")
            # per-device empty cache
            for did in range(layer.num_devices):
                with torch.cuda.device(did):
                    torch.cuda.empty_cache()
            for did in range(layer.num_devices):
                with torch.cuda.device(did):
                    print(f"After empty  dev{did}: allocated={torch.cuda.memory_allocated(did)/1e9:.2f} GB  reserved={torch.cuda.memory_reserved(did)/1e9:.2f} GB")

    layer.cleanup()

if __name__ == "__main__":
    main()
