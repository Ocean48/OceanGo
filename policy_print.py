import argparse
import os
import numpy as np
import torch

from OceanGo import (
    IN_PLANES,
    PolicyValueNet,
    load_checkpoint_auto,
    resolve_latest_v2_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect OceanGo V2 GlobalPoolResNet checkpoint & parameters")
    parser.add_argument("--model", type=str, default="", help="Path to V2 .pth checkpoint (default: auto-detect latest)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory to search")
    args = parser.parse_args()

    model_file = args.model if args.model else resolve_latest_v2_checkpoint(args.save_dir)
    if not model_file or not os.path.isfile(model_file):
        print("No V2 checkpoint found yet. Constructing default 15x192 PolicyValueNet for architectural inspection...\n")
        net = PolicyValueNet(board_size=9)
        meta = {
            "version": 2,
            "arch": "GlobalPoolResNet (untrained)",
            "blocks": net.num_res_blocks,
            "channels": net.num_channels,
            "board_size": 9,
            "iteration": 0,
            "elo_est": 1000.0,
        }
    else:
        meta = load_checkpoint_auto(model_file)
        if meta.get("version") != 2:
            print(f"Checkpoint '{model_file}' is a legacy V1 fixed-board file (superseded by V2 GlobalPoolResNet).")
            return
        net = PolicyValueNet(
            board_size=meta.get("board_size", 9),
            num_channels=meta["channels"],
            num_res_blocks=meta["blocks"],
        )
        net.load_state_dict(meta["state_dict"], strict=True)
        print(f"==> Loaded V2 checkpoint '{model_file}' successfully!")

    net.eval()
    total_params = sum(p.numel() for p in net.parameters())

    print("========================================")
    print(f"Architecture : {meta.get('arch', 'GlobalPoolResNet')} (V{meta.get('version', 2)})")
    print(f"Backbone     : {meta['blocks']} blocks x {meta['channels']} channels")
    print(f"Trained Size : {meta.get('board_size', 9)}x{meta.get('board_size', 9)}")
    print(f"Iteration    : {meta.get('iteration', 0)} (Est. Elo: {meta.get('elo_est', 1000.0):.0f})")
    print(f"Total Params : {total_params:,}")
    print("========================================\n")

    # Verify board-size-agnostic inference across 9x9, 13x13, and 19x19 with the exact same weights
    print("==> Verifying 100% Board-Size-Agnostic Inference (9x9 -> 13x13 -> 19x19):")
    with torch.no_grad():
        for b_sz in (9, 13, 19):
            dummy = torch.zeros((1, IN_PLANES, b_sz, b_sz), dtype=torch.float32)
            dummy[:, 17, :, :] = 1.0
            p_logits, v_logit, own_logits = net(dummy, return_aux=True)
            v_val = float(torch.tanh(0.5 * v_logit).item())
            print(
                f"  - {b_sz:2d}x{b_sz:<2d}: policy={tuple(p_logits.shape)}, "
                f"value={v_val:+.4f}, ownership={tuple(own_logits.shape)}"
            )
    print()

    print("==> Network Parameter Summary:\n")
    for name, param in net.named_parameters():
        data_np = param.detach().cpu().numpy()
        print(
            f"  {name:<36} shape={str(tuple(param.size())):<20} "
            f"mean={np.mean(data_np):+.5f}  std={np.std(data_np):.5f}"
        )

    print("\n==> Done inspecting OceanGo V2 model parameters.")


if __name__ == "__main__":
    main()
