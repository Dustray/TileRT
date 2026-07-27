"""Test NCCL multi-process all-reduce across 8 DCUs."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    t = torch.ones(4, device=f"cuda:{rank}") * (rank + 1)
    dist.all_reduce(t)
    print(f"rank {rank} result: {t.tolist()}", flush=True)
    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    print(f"world_size={world_size}")
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
