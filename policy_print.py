import argparse
import os
import numpy as np
import torch

from OceanGo import (
    GoGame,
    IN_PLANES,
    PolicyValueNet,
    is_state_dict_finite,
    load_checkpoint_auto,
    resolve_latest_v2_checkpoint,
    softmax,
)


def scan_checkpoint_dir(save_dir: str):
    if not os.path.isdir(save_dir):
        return
    pth_files = sorted(f for f in os.listdir(save_dir) if f.endswith(".pth") and not f.startswith("train_state"))
    if not pth_files:
        return
    print(f"==> Scanning Checkpoints in '{save_dir}':")
    for fname in pth_files:
        fpath = os.path.join(save_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            meta = load_checkpoint_auto(fpath)
        except Exception as e:
            print(f"  - {fname:<34} : UNREADABLE ({e})")
            continue
        if meta.get("version") != 2:
            print(f"  - {fname:<34} : Legacy V1 checkpoint (skipped by V2)")
            continue
        finite_ok = is_state_dict_finite(meta.get("state_dict"))
        status = "HEALTHY (finite)" if finite_ok else "CORRUPTED (NaN/Inf weights!)"
        b_sz = meta.get("board_size", 9)
        it_n = meta.get("iteration", 0)
        elo = meta.get("elo_est", 1000.0)
        print(f"  - {fname:<34} : {status:<28} | {b_sz}x{b_sz} | iter={it_n:<4} | Elo={elo:.0f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect OceanGo V2 GlobalPoolResNet checkpoint & parameters")
    parser.add_argument("--model", type=str, default="", help="Path to V2 .pth checkpoint (default: auto-detect latest)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory to search")
    args = parser.parse_args()

    scan_checkpoint_dir(args.save_dir)

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
        sd = meta["state_dict"]
        if not is_state_dict_finite(sd):
            bad_keys = [k for k, v in sd.items() if isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item()]
            print(f"[ERROR] Checkpoint '{model_file}' contains CORRUPTED (NaN/Inf) tensors in {len(bad_keys)}/{len(sd)} entries!")
            print(f"        First corrupted tensors: {bad_keys[:5]}")
            print("        Cause: Blackwell sm_120 cuDNN NHWC bfloat16 kernel NaN or unguarded gradient Inf.")
            print("        Fix: Use the latest HEALTHY checkpoint listed above, or retrain with the updated OceanGo.py.\n")
            return
        net = PolicyValueNet(
            board_size=meta.get("board_size", 9),
            num_channels=meta["channels"],
            num_res_blocks=meta["blocks"],
        )
        net.load_state_dict(sd, strict=True)
        print(f"==> Loaded V2 checkpoint '{model_file}' successfully (all weights & BatchNorm stats finite)!")

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
    print("==> Verifying 100% Board-Size-Agnostic Inference & Opening Pass Priors (9x9 -> 13x13 -> 19x19):")
    with torch.no_grad():
        for b_sz in (9, 13, 19):
            g_empty = GoGame(b_sz)
            inp_empty = torch.from_numpy(g_empty.get_nn_input()[None])
            p_logits, v_logit, own_logits = net(inp_empty, return_aux=True)
            probs = softmax(p_logits[0].numpy())
            p_pass = float(probs[-1])
            board_probs = probs[:-1]
            top_idx = int(np.argmax(board_probs))
            top_mv = (top_idx // b_sz, top_idx % b_sz)
            v_val = float(torch.tanh(0.5 * v_logit).item())
            print(
                f"  - {b_sz:2d}x{b_sz:<2d}: policy={tuple(p_logits.shape)}, "
                f"value={v_val:+.4f}, P(Pass|Move0)={p_pass*100:5.2f}%, "
                f"TopBoard={top_mv} ({board_probs[top_idx]*100:5.2f}%), ownership={tuple(own_logits.shape)}"
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
