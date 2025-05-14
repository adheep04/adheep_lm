import model
import torch
from dataclasses import dataclass


def main():
    args = model.ModelArgs()
    x = torch.randn(8, 5, args.d_model)
    mask = torch.randn(8, 5, args.d_model)
    rot_factors = torch.randn(12)
    attn_block = model.MultiheadLatentAttn(args, mask)

    out = attn_block(x, 0, rot_factors, mask)
    print(x.size())


if __name__ == "__main__":
    main()

