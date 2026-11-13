#!/usr/bin/env python3
"""
OceanGo V2 — Multi-Machine IEEE Paper Benchmark & Telemetry Extractor
=====================================================================

Generates empirical tables and training telemetry for Section VI of:
  "Compute-Efficient AlphaGo: Board-Agnostic Curriculum Learning and
   Monte Carlo Tree Search for Single-Machine Self-Play Reinforcement Learning"

Suites:
  1. hw       -> Table II:  Hardware profile, OS, CPU, RAM, accelerator, precision, params
  2. engine   -> Table III: Legacy V1 (deepcopy+DFS) vs V2 (Zobrist+Bitmask) & RAM compression
  3. ipc      -> Table IV:  Pipe-Pickled Queue vs Zero-Pickle SharedMemory + Playout Caps & SGD speed
  4. transfer -> Table V:   Zero-shot cross-board curriculum transfer (9x9 -> 13x13 -> 19x19)
  5. arena    -> Table VI:  Test-time 8x D4 symmetry ensembling & subtree reuse Arena strength
  6. logs     -> Sec VI-C:  Structured parser for curriculum_console.log & logs_*.log

Usage:
  python benchmark_paper.py --quick          # Fast 1-2 min smoke verification
  python benchmark_paper.py                  # Full paper-grade benchmark across all suites
  python benchmark_paper.py --suite transfer # Run a specific suite only
"""

import argparse
import copy
import datetime
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import random
import re
import subprocess
import sys
import threading
import time
import concurrent.futures
import numpy as np
import torch

import OceanGo as og
import OceanGo_legacy as og_legacy


# ------------------------------------------------------------------------------
# Top-Level Spawn-Safe Worker Initializer for Pipe-Pickled Queue Baseline
# ------------------------------------------------------------------------------
def _pipe_queue_worker_init(req_queue, res_queues, progress_queue, slot_batch, board_size, force_uniform_budget=True):
    """
    Initializes a worker WITHOUT SharedMemory buffers so OceanGo falls back
    to standard pickled NumPy arrays over multiprocessing.Queue.
    """
    if force_uniform_budget:
        og.FULL_SEARCH_PROB = 1.0  # Baseline without playout cap oscillations
    og._init_self_play_worker(
        req_queue=req_queue,
        res_queues=res_queues,
        progress_queue=progress_queue,
        shm_req_name=None,
        shm_res_name=None,
        num_workers=0,
        slot_batch=slot_batch,
        board_size=board_size,
    )


# ------------------------------------------------------------------------------
# Platform, Hardware & Checkpoint Discovery Utilities
# ------------------------------------------------------------------------------
def detect_cpu_brand() -> str:
    sys_name = platform.system().lower()
    try:
        if sys_name == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
            if out:
                return out
        elif sys_name == "linux" and os.path.isfile("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        _, val = line.split(":", 1)
                        val = val.strip()
                        if val:
                            return val
        elif sys_name == "windows":
            import winreg  # type: ignore
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if val and str(val).strip():
                    return str(val).strip()
    except Exception:
        pass
    proc = platform.processor().strip()
    return proc if proc else platform.machine()


def make_machine_slug(cpu_brand: str) -> str:
    if og.device.type == "cuda":
        raw = og.device_name.split("(")[0].strip()
    elif og.device.type == "mps":
        raw = cpu_brand
    elif og.IS_TPU:
        raw = og.device_name
    else:
        raw = f"{platform.system()}_{cpu_brand}"
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return slug[:48] if slug else "unknown_machine"


def _elo_from_win_rate(win_rate: float) -> float:
    clamped = min(0.975, max(0.025, float(win_rate)))
    return 400.0 * math.log10(clamped / (1.0 - clamped))


def _prep_net_for_device(net: torch.nn.Module) -> torch.nn.Module:
    net = net.to(device=og.device)
    if og.USE_CHANNELS_LAST:
        net = net.to(memory_format=torch.channels_last)
    net.eval()
    return net


def discover_checkpoints(save_dir: str) -> dict:
    """
    Scans `save_dir` for finite V2 checkpoints grouped by board size (9, 13, 19).
    Prefers `policy_value_net_best_{N}x{N}.pth`, falling back to highest-iteration
    `policy_value_net_iter_*.pth` on that board size. Tracks `dry_run_test.pth`
    separately so untrained dry-run files are never mistaken for trained models.
    """
    info = {
        "trained_by_board": {},
        "all_valid_v2": [],
        "dry_run_fallback": None,
        "detected_blocks": None,
        "detected_channels": None,
    }
    search_dirs = []
    if save_dir and os.path.isdir(save_dir):
        search_dirs.append(save_dir)
    if os.path.normpath(save_dir or "") in ("checkpoints", ".", "") and os.path.isdir("."):
        if os.path.abspath(".") != os.path.abspath(save_dir or "."):
            search_dirs.append(".")

    seen_paths = set()
    best_by_board = {}
    latest_by_board = {}

    for d in search_dirs:
        try:
            files = sorted(os.listdir(d))
        except OSError:
            continue
        for fname in files:
            if not fname.endswith(".pth") or fname.startswith("train_state"):
                continue
            fpath = os.path.join(d, fname)
            ap = os.path.abspath(fpath)
            if not os.path.isfile(fpath) or ap in seen_paths:
                continue
            seen_paths.add(ap)
            try:
                meta = og.load_checkpoint_auto(fpath)
            except Exception:
                continue
            if meta.get("version") != 2 or not og.is_state_dict_finite(meta.get("state_dict")):
                continue

            entry = {
                "path": fpath,
                "filename": fname,
                "board_size": int(meta.get("board_size", 9)),
                "blocks": int(meta.get("blocks", og.RESNET_BLOCKS)),
                "channels": int(meta.get("channels", og.RESNET_CHANNELS)),
                "iteration": int(meta.get("iteration", 0)),
                "stage_iter": int(meta.get("stage_iter", meta.get("iteration", 0))),
                "elo_est": float(meta.get("elo_est", 1000.0)),
                "mtime": os.path.getmtime(fpath),
            }
            if fname == "dry_run_test.pth":
                info["dry_run_fallback"] = entry
                continue

            info["all_valid_v2"].append(entry)
            b = entry["board_size"]
            if fname == f"policy_value_net_best_{b}x{b}.pth":
                best_by_board[b] = entry
            key = (entry["iteration"], entry["stage_iter"], entry["mtime"])
            prev = latest_by_board.get(b)
            if prev is None or key > (prev["iteration"], prev["stage_iter"], prev["mtime"]):
                latest_by_board[b] = entry

    for b in (9, 13, 19):
        if b in best_by_board:
            info["trained_by_board"][b] = best_by_board[b]
        elif b in latest_by_board:
            info["trained_by_board"][b] = latest_by_board[b]

    # Pick architecture from highest board/iteration checkpoint
    all_cands = list(info["trained_by_board"].values()) or info["all_valid_v2"]
    if all_cands:
        top = max(all_cands, key=lambda e: (e["board_size"], e["iteration"], e["mtime"]))
        info["detected_blocks"] = top["blocks"]
        info["detected_channels"] = top["channels"]

    return info


def detect_arch_from_logs(log_files: list) -> tuple:
    for lf in log_files:
        if not lf or not os.path.isfile(lf):
            continue
        try:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(8192)
            m = re.search(r"(\d+)\s*blocks\s*x\s*(\d+)\s*channels", head)
            if m:
                return int(m.group(1)), int(m.group(2))
        except OSError:
            pass
    return None, None


# ------------------------------------------------------------------------------
# Suite 1: Hardware & Model Profile (Table II)
# ------------------------------------------------------------------------------
def benchmark_hardware_profile(blocks: int, channels: int, workers: int) -> dict:
    cpu_brand = detect_cpu_brand()
    sys_cores = os.cpu_count() or 1
    amp_str = str(og.AMP_DTYPE).replace("torch.", "") if og.AMP_ENABLED else "float32"
    layout_str = "channels_last" if og.USE_CHANNELS_LAST else "contiguous"

    param_counts = {}
    for b_cnt, c_cnt in ((15, 192), (18, 256), (blocks, channels)):
        key = f"{b_cnt}x{c_cnt}"
        if key not in param_counts:
            tmp_net = og.PolicyValueNet(board_size=9, num_channels=c_cnt, num_res_blocks=b_cnt)
            param_counts[key] = sum(p.numel() for p in tmp_net.parameters())

    profile = {
        "os": f"{platform.system()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_brand": cpu_brand,
        "cpu_logical_threads": sys_cores,
        "workers_used": workers,
        "system_ram_gb": round(float(og.SYSTEM_RAM_GB), 2),
        "device_type": og.device.type,
        "device_name": og.device_name,
        "gpu_vram_gb": round(float(og.GPU_VRAM_GB), 2),
        "precision": amp_str,
        "memory_layout": layout_str,
        "active_backbone": f"{blocks}x{channels}",
        "active_params": param_counts[f"{blocks}x{channels}"],
        "param_counts": param_counts,
        "machine_slug": make_machine_slug(cpu_brand),
    }
    return profile


# ------------------------------------------------------------------------------
# Suite 2: Legacy V1 vs OceanGo V2 Rules Engine & Compression (Table III)
# ------------------------------------------------------------------------------
def _time_legacy_v1_engine(board_size: int, target_plies: int, seed: int) -> dict:
    rng = random.Random(seed)
    total_plies = 0
    games_started = 0
    g = og_legacy.GoGame(board_size)
    games_started += 1
    t0 = time.perf_counter()

    while total_plies < target_plies:
        legals = g.get_legal_moves()
        _ = g.get_state()  # V1 deep-copy clone cost in MCTS expansion
        if len(legals) <= max(2, board_size // 3):
            g = og_legacy.GoGame(board_size)
            games_started += 1
            continue
        mv = rng.choice(legals)
        g.make_move(mv[0], mv[1])
        total_plies += 1

    elapsed = max(1e-6, time.perf_counter() - t0)
    return {
        "plies": total_plies,
        "games": games_started,
        "elapsed_sec": elapsed,
        "ply_per_sec": total_plies / elapsed,
    }


def _time_v2_engine(board_size: int, target_plies: int, seed: int) -> dict:
    rng = random.Random(seed)
    total_plies = 0
    games_started = 0
    g = og.GoGame(board_size)
    games_started += 1
    t0 = time.perf_counter()

    while total_plies < target_plies:
        legals = g.get_legal_moves()
        _ = g.copy()  # V2 shallow immutable chain-slice copy
        board_legals = [mv for mv in legals if mv != (-1, -1)]
        if len(board_legals) <= max(2, board_size // 3):
            g = og.GoGame(board_size)
            games_started += 1
            continue
        mv = rng.choice(board_legals)
        g.make_move(mv[0], mv[1], check_valid=True)
        total_plies += 1

    elapsed = max(1e-6, time.perf_counter() - t0)
    return {
        "plies": total_plies,
        "games": games_started,
        "elapsed_sec": elapsed,
        "ply_per_sec": total_plies / elapsed,
    }


def benchmark_engine_v1_vs_v2(quick: bool = False) -> list:
    ply_targets = {9: 100, 13: 60, 19: 40} if quick else {9: 300, 13: 180, 19: 120}
    results = []
    for n in (9, 13, 19):
        plies = ply_targets[n]
        v1 = _time_legacy_v1_engine(n, plies, seed=1000 + n)
        v2 = _time_v2_engine(n, plies, seed=1000 + n)
        raw_bytes = 18 * n * n * 4
        packed_bytes = n * n * 2
        # CompactReplayBuffer full per-sample footprint:
        # packed_u16(2N^2) + cplayer(1) + pi_fp16(2*(N^2+1)) + z_fp16(2) + pw_fp16(2) + own_u8(N^2) + ow_fp16(2)
        full_tuple_bytes = (2 * n * n) + 1 + 2 * (n * n + 1) + 2 + 2 + (n * n) + 2
        ram_500k_gb = (500_000 * full_tuple_bytes) / 1e9  # Decimal GB matching Table III & Sec III-C
        results.append({
            "board_size": n,
            "target_plies": plies,
            "v1_ply_per_sec": v1["ply_per_sec"],
            "v2_ply_per_sec": v2["ply_per_sec"],
            "speedup": v2["ply_per_sec"] / max(1e-6, v1["ply_per_sec"]),
            "raw_float32_bytes": raw_bytes,
            "packed_uint16_bytes": packed_bytes,
            "state_compression": raw_bytes / packed_bytes,
            "full_tuple_bytes": full_tuple_bytes,
            "replay_500k_gb": ram_500k_gb,
        })
    return results


# ------------------------------------------------------------------------------
# Suite 3: Parallel IPC (Pipe Queue vs SharedMemory + Playout Caps) & SGD (Table IV)
# ------------------------------------------------------------------------------
def _run_pipe_queue_self_play_batch(net, board_size: int, n_games: int, n_sims: int, num_workers: int, mp_ctx, force_uniform_budget: bool = True):
    num_workers = max(1, min(num_workers, n_games))
    req_queue = mp_ctx.Queue()
    res_queues = [mp_ctx.Queue() for _ in range(num_workers)]
    progress_queue = mp_ctx.Queue()

    gpu_thread = threading.Thread(
        target=og.gpu_server_thread,
        args=(net, req_queue, res_queues, num_workers, None, None, board_size),
        daemon=True,
    )
    gpu_thread.start()

    totals = {"games": 0, "total_moves": 0}
    eval_counter = [0]
    hook = og.unwrap_model(net).register_forward_hook(
        lambda _m, inp, _out: eval_counter.__setitem__(0, eval_counter[0] + int(inp[0].shape[0]))
    )

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp_ctx,
            initializer=_pipe_queue_worker_init,
            initargs=(req_queue, res_queues, progress_queue, og.MCTS_BATCH_SIZE, board_size, force_uniform_budget),
        ) as executor:
            futures = []
            base_games, rem_games = divmod(n_games, num_workers)
            start_t = time.perf_counter()
            for w_id in range(num_workers):
                g_cnt = base_games + (1 if w_id < rem_games else 0)
                if g_cnt > 0:
                    futures.append(executor.submit(og.self_play_wrapper, board_size, n_sims, w_id, g_cnt))
                else:
                    req_queue.put("DONE")

            for f in futures:
                _states, w_stats = f.result()
                totals["games"] += w_stats["games"]
                totals["total_moves"] += w_stats["total_moves"]
                req_queue.put("DONE")
            elapsed = max(1e-6, time.perf_counter() - start_t)
        gpu_thread.join(timeout=10.0)
    finally:
        hook.remove()

    totals["elapsed_sec"] = elapsed
    totals["nn_evals"] = eval_counter[0]
    return totals


def _run_shm_timed_self_play_batch(net, board_size: int, n_games: int, n_sims: int, num_workers: int, mp_ctx):
    eval_counter = [0]
    hook = og.unwrap_model(net).register_forward_hook(
        lambda _m, inp, _out: eval_counter.__setitem__(0, eval_counter[0] + int(inp[0].shape[0]))
    )
    try:
        t0 = time.perf_counter()
        _states, sp_stats = og.run_parallel_self_play_batch(
            net=net,
            board_size=board_size,
            n_games=n_games,
            n_sims=n_sims,
            num_workers=num_workers,
            mp_ctx=mp_ctx,
            show_progress=False,
        )
        elapsed = max(1e-6, time.perf_counter() - t0)
    finally:
        hook.remove()

    # Use inner elapsed_sec from run_parallel_self_play_batch (excluding process spawn) when available, and also track wall
    inner_elapsed = max(1e-6, float(sp_stats.get("elapsed_sec", elapsed)))
    return {
        "games": int(sp_stats["games"]),
        "total_moves": int(sp_stats["total_moves"]),
        "elapsed_sec": inner_elapsed,
        "wall_elapsed_sec": elapsed,
        "nn_evals": eval_counter[0],
        "resigned_games": int(sp_stats.get("resigned_games", 0)),
    }


def _benchmark_sgd_speed(base_net: torch.nn.Module, board_size: int, batch_size: int, warmup_steps: int, bench_steps: int) -> dict:
    """Measures steady-state SGD + on-GPU 8x D4 symmetry training throughput on a cloned net."""
    net_clone = _prep_net_for_device(copy.deepcopy(og.unwrap_model(base_net)))
    buf_cap = max(batch_size * 4, 1024)
    buf = og.CompactReplayBuffer(max_size=buf_cap, board_size=board_size)

    # Populate synthetic legal Go states
    rng = random.Random(2026 + board_size)
    samples = []
    g = og.GoGame(board_size)
    target_unique = min(256, buf_cap)
    pi_uniform = np.full(board_size * board_size + 1, 1.0 / (board_size * board_size + 1), dtype=np.float32)
    own_sample = np.ones((board_size, board_size), dtype=np.uint8)
    for _ in range(target_unique):
        pk, cp = g.get_packed_state()
        samples.append((pk, cp, pi_uniform, 1.0, 1.0, own_sample, 1.0))
        legals = [mv for mv in g.get_legal_moves() if mv != (-1, -1)]
        if len(legals) < 4:
            g = og.GoGame(board_size)
        else:
            g.make_move(*rng.choice(legals), check_valid=False)
    while len(buf) < buf_cap:
        buf.extend(samples)

    opt = og.build_optimizer(net_clone, lr=0.01)

    # Suppress verbose stdout progress bar during timed steps
    orig_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            og.train_policy_value_net(net_clone, replay_buffer=buf, batch_size=batch_size, lr=0.01, optimizer=opt, num_steps=warmup_steps)
            if og.device.type == "cuda":
                torch.cuda.synchronize()
            elif og.device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            og.train_policy_value_net(net_clone, replay_buffer=buf, batch_size=batch_size, lr=0.01, optimizer=opt, num_steps=bench_steps)
            if og.device.type == "cuda":
                torch.cuda.synchronize()
            elif og.device.type == "mps":
                torch.mps.synchronize()
            dt = max(1e-6, time.perf_counter() - t0)
    finally:
        sys.stdout = orig_stdout
        del net_clone, opt, buf

    return {
        "batch_size": batch_size,
        "steps": bench_steps,
        "elapsed_sec": dt,
        "steps_per_sec": bench_steps / dt,
        "samples_per_sec": (bench_steps * batch_size) / dt,
    }


def benchmark_ipc_and_training(ckpt_info: dict, blocks: int, channels: int, workers: int, mp_ctx, quick: bool = False) -> list:
    trained_map = ckpt_info["trained_by_board"]
    fallback_entry = ckpt_info.get("dry_run_fallback")

    def _get_net_for_board(b_sz: int):
        # Prefer trained checkpoint matching board_size (or any trained checkpoint via 100% weight transfer)
        src = trained_map.get(b_sz)
        if src is None and trained_map:
            src = max(trained_map.values(), key=lambda e: (e["board_size"], e["iteration"]))
        if src is None:
            src = fallback_entry
        if src is not None:
            meta = og.load_checkpoint_auto(src["path"])
            net = og.PolicyValueNet(board_size=b_sz, num_channels=meta["channels"], num_res_blocks=meta["blocks"])
            net.load_state_dict(meta["state_dict"], strict=True)
            note = f"{src['filename']} (iter={src['iteration']})"
        else:
            net = og.PolicyValueNet(board_size=b_sz, num_channels=channels, num_res_blocks=blocks)
            note = "fresh_init"
        return _prep_net_for_device(net), note

    # In quick mode, run full Playout Cap (>=120 sims) on 9x9 and a fast IPC smoke check on 19x19
    if quick:
        configs = [
            {"board_size": 9, "n_games": min(workers, 4), "n_sims": 120, "sgd_bs": 128, "sgd_steps": 10},
            {"board_size": 19, "n_games": min(workers, 2), "n_sims": 16, "sgd_bs": 128, "sgd_steps": 8},
        ]
    else:
        configs = [
            {"board_size": 9, "n_games": max(16, workers), "n_sims": 600, "sgd_bs": 256, "sgd_steps": 40},
            {"board_size": 19, "n_games": max(8, workers), "n_sims": 600, "sgd_bs": 256, "sgd_steps": 30},
        ]

    rows = []
    for cfg in configs:
        b_sz = cfg["board_size"]
        n_games = cfg["n_games"]
        n_sims = cfg["n_sims"]
        net, net_note = _get_net_for_board(b_sz)

        # Warm up 1 tiny inference pass so lazy MPS/CUDA contexts aren't counted
        dummy = np.zeros((2, og.IN_PLANES, b_sz, b_sz), dtype=np.uint8)
        _ = og.forward_infer_batch(net, dummy)

        print(f"    -> [{b_sz}x{b_sz}] Pipe-Pickled Queue ({n_games}g @ {n_sims} sims, {min(workers, n_games)}W, net={net_note})...", flush=True)
        pipe_res = _run_pipe_queue_self_play_batch(net, b_sz, n_games, n_sims, workers, mp_ctx, force_uniform_budget=True)

        print(f"    -> [{b_sz}x{b_sz}] Zero-Pickle SharedMemory + Playout Caps ({n_games}g @ {n_sims} sims)...", flush=True)
        shm_res = _run_shm_timed_self_play_batch(net, b_sz, n_games, n_sims, workers, mp_ctx)

        print(f"    -> [{b_sz}x{b_sz}] SGD Train Benchmark (B={cfg['sgd_bs']}, {cfg['sgd_steps']} steps)...", flush=True)
        sgd_res = _benchmark_sgd_speed(net, b_sz, batch_size=cfg["sgd_bs"], warmup_steps=2, bench_steps=cfg["sgd_steps"])

        pipe_gps = pipe_res["games"] / pipe_res["elapsed_sec"]
        shm_gps = shm_res["games"] / shm_res["elapsed_sec"]
        pipe_mps = pipe_res["total_moves"] / pipe_res["elapsed_sec"]
        shm_mps = shm_res["total_moves"] / shm_res["elapsed_sec"]
        nn_evals_per_sec = shm_res["nn_evals"] / shm_res["elapsed_sec"]
        effective_sims_per_sec = (shm_res["total_moves"] * n_sims) / shm_res["elapsed_sec"]

        rows.append({
            "board_size": b_sz,
            "n_games": n_games,
            "n_sims_full": n_sims,
            "workers": min(workers, n_games),
            "model_source": net_note,
            "pipe_games_per_sec": pipe_gps,
            "pipe_moves_per_sec": pipe_mps,
            "pipe_nn_evals_per_sec": pipe_res["nn_evals"] / pipe_res["elapsed_sec"],
            "pipe_elapsed_sec": pipe_res["elapsed_sec"],
            "shm_games_per_sec": shm_gps,
            "shm_moves_per_sec": shm_mps,
            "shm_nn_evals_per_sec": nn_evals_per_sec,
            "shm_effective_sims_per_sec": effective_sims_per_sec,
            "shm_elapsed_sec": shm_res["elapsed_sec"],
            "games_speedup": shm_gps / max(1e-6, pipe_gps),
            "moves_speedup": shm_mps / max(1e-6, pipe_mps),
            "sgd_batch_size": sgd_res["batch_size"],
            "sgd_steps_per_sec": sgd_res["steps_per_sec"],
            "sgd_samples_per_sec": sgd_res["samples_per_sec"],
        })
        del net
    return rows


# ------------------------------------------------------------------------------
# Custom Lockstep Arena Supporting Symmetry Count, Subtree Reuse & Timing
# ------------------------------------------------------------------------------
def run_custom_arena(
    cand_net: torch.nn.Module,
    ref_net: torch.nn.Module,
    board_size: int,
    num_games: int = 20,
    cand_sims: int = 160,
    ref_sims: int = 160,
    cand_symmetries: int = 1,
    ref_symmetries: int = 1,
    cand_reuse: bool = True,
    ref_reuse: bool = True,
    intra_batch: int = 8,
    max_moves_override: int = None,
) -> dict:
    """
    Runs paired-opening synchronous lockstep Arena evaluation with independent
    simulation budgets, D4 symmetry ensembling, subtree reuse toggles, and
    per-move latency telemetry.
    """
    cand_net.eval()
    ref_net.eval()

    n_pairs = max(1, num_games // 2)
    total_games = n_pairs * 2
    openings = og._generate_varied_2move_openings(board_size, num_pairs=n_pairs)
    max_moves = max_moves_override if max_moves_override is not None else (board_size * board_size * 2)

    games = []
    cand_colors = []
    roots_b = []
    roots_w = []
    active = []
    move_counts = []

    # Separate trees per side whenever subtree reuse differs or trees aren't shared cleanly
    for m1, m2 in openings:
        for cand_col in (1, 2):
            g = og.GoGame(board_size)
            g.make_move(*m1, check_valid=True)
            g.make_move(*m2, check_valid=True)
            games.append(g)
            cand_colors.append(cand_col)
            # In self-play / asymmetric Arena, each player maintains its own search root cleanly
            roots_b.append(og.NN_MCTS_Node(g.copy()))
            roots_w.append(og.NN_MCTS_Node(g.copy()))
            active.append(True)
            move_counts.append(2)

    def _batch_eval_for_model(model, node_list, n_sym: int):
        if not node_list:
            return
        eval_nds = []
        for nd in node_list:
            if nd.is_expanded:
                continue
            nd.ensure_game_state()
            if nd.game_state.is_game_over():
                w = nd.game_state.get_winner()
                nd.value = 0.0 if w == 0 else (1.0 if w == nd.game_state.current_player else -1.0)
                nd.is_expanded = True
            else:
                eval_nds.append(nd)
        if not eval_nds:
            return
        b_arr = np.empty((len(eval_nds), og.IN_PLANES, board_size, board_size), dtype=np.uint8)
        for i, nd in enumerate(eval_nds):
            nd.game_state.write_nn_input_uint8(b_arr[i])
        if n_sym > 1:
            p_logits, v_out = og.ensemble_infer_symmetries(model, b_arr.astype(np.float32), num_symmetries=n_sym)
        else:
            p_logits, v_out = og.forward_infer_batch(model, b_arr)
        pass_idx = board_size * board_size
        for i, nd in enumerate(eval_nds):
            nd.value = float(v_out[i])
            legals = nd.game_state.get_legal_moves()
            m_idxs = [(pass_idx if mv == (-1, -1) else mv[0] * board_size + mv[1]) for mv in legals]
            probs = og.softmax(p_logits[i, m_idxs]) if m_idxs else ()
            for mv, pr in zip(legals, probs):
                nd.children[mv] = og.NN_MCTS_Node(game_state=None, parent=nd, move=mv, prior=float(pr))
            nd.is_expanded = True

    cand_time_sec = 0.0
    ref_time_sec = 0.0
    cand_moves = 0
    ref_moves = 0

    while any(active):
        active_indices = [idx for idx, is_act in enumerate(active) if is_act]
        active_roots = {idx: (roots_b[idx] if games[idx].current_player == 1 else roots_w[idx]) for idx in active_indices}

        cand_unexp = []
        ref_unexp = []
        for idx in active_indices:
            rt = active_roots[idx]
            if not rt.is_expanded:
                if games[idx].current_player == cand_colors[idx]:
                    cand_unexp.append(rt)
                else:
                    ref_unexp.append(rt)

        t0 = time.perf_counter()
        _batch_eval_for_model(cand_net, cand_unexp, cand_symmetries)
        cand_time_sec += time.perf_counter() - t0

        t0 = time.perf_counter()
        _batch_eval_for_model(ref_net, ref_unexp, ref_symmetries)
        ref_time_sec += time.perf_counter() - t0

        for rt in cand_unexp + ref_unexp:
            if rt.N == 0:
                rt.N = 1
                rt.W = -rt.value
                rt.Q = -rt.value

        sims_done = {idx: 0 for idx in active_indices}
        sims_target = {idx: (cand_sims if games[idx].current_player == cand_colors[idx] else ref_sims) for idx in active_indices}

        while True:
            need_sim = [idx for idx in active_indices if sims_done[idx] < sims_target[idx]]
            if not need_sim:
                break

            cand_leaves, cand_paths = [], []
            ref_leaves, ref_paths = [], []
            game_step_counts = {idx: 0 for idx in need_sim}

            cand_sel_t = 0.0
            ref_sel_t = 0.0
            for idx in need_sim:
                t_s = time.perf_counter()
                rt = active_roots[idx]
                is_cand_turn = (games[idx].current_player == cand_colors[idx])
                seen_local = set()
                cur_b = min(intra_batch, sims_target[idx] - sims_done[idx])
                for _ in range(cur_b):
                    leaf, path = og.mcts_select(rt)
                    if leaf in seen_local:
                        break
                    seen_local.add(leaf)
                    og.apply_virtual_loss(path)
                    if is_cand_turn:
                        cand_leaves.append(leaf)
                        cand_paths.append(path)
                    else:
                        ref_leaves.append(leaf)
                        ref_paths.append(path)
                    game_step_counts[idx] += 1
                if is_cand_turn:
                    cand_sel_t += time.perf_counter() - t_s
                else:
                    ref_sel_t += time.perf_counter() - t_s
            cand_time_sec += cand_sel_t
            ref_time_sec += ref_sel_t

            t0 = time.perf_counter()
            _batch_eval_for_model(cand_net, cand_leaves, cand_symmetries)
            for path, leaf in zip(cand_paths, cand_leaves):
                og.remove_virtual_loss(path)
                og.mcts_backup(path, leaf.value)
            cand_time_sec += time.perf_counter() - t0

            t0 = time.perf_counter()
            _batch_eval_for_model(ref_net, ref_leaves, ref_symmetries)
            for path, leaf in zip(ref_paths, ref_leaves):
                og.remove_virtual_loss(path)
                og.mcts_backup(path, leaf.value)
            ref_time_sec += time.perf_counter() - t0

            for idx in need_sim:
                sims_done[idx] += max(1, game_step_counts[idx])

        for idx in active_indices:
            g = games[idx]
            is_cand_turn = (g.current_player == cand_colors[idx])
            rt = active_roots[idx]
            if not rt.children:
                active[idx] = False
                continue
            if move_counts[idx] < 6:
                pi_d = og.extract_mcts_policy_target(rt, prune_single_visit_noise=False)
                mv = og.sample_move_with_temperature(pi_d, move_count=move_counts[idx], board_size=board_size, greedy=False)
            else:
                mv = max(rt.children.items(), key=lambda kv: kv[1].N)[0]
            g.make_move(*mv, check_valid=False)
            move_counts[idx] += 1
            if is_cand_turn:
                cand_moves += 1
            else:
                ref_moves += 1

            if g.is_game_over() or move_counts[idx] >= max_moves:
                active[idx] = False
            else:
                # Advance Black's and White's own trees independently so asymmetric reuse flags work properly
                cand_is_b = (cand_colors[idx] == 1)
                reuse_b = cand_reuse if cand_is_b else ref_reuse
                reuse_w = ref_reuse if cand_is_b else cand_reuse
                roots_b[idx] = og.advance_mcts_root(roots_b[idx], mv, g) if reuse_b else og.NN_MCTS_Node(g.copy())
                roots_w[idx] = og.advance_mcts_root(roots_w[idx], mv, g) if reuse_w else og.NN_MCTS_Node(g.copy())

    wins, losses, draws = 0, 0, 0
    for idx, g in enumerate(games):
        w = g.get_winner()
        if w == 0:
            draws += 1
        elif w == cand_colors[idx]:
            wins += 1
        else:
            losses += 1

    win_rate = (wins + 0.5 * draws) / float(total_games)
    elo_diff = _elo_from_win_rate(win_rate)
    return {
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "elo_diff": elo_diff,
        "cand_avg_ms_per_move": (cand_time_sec * 1000.0) / max(1, cand_moves),
        "ref_avg_ms_per_move": (ref_time_sec * 1000.0) / max(1, ref_moves),
        "cand_moves": cand_moves,
        "ref_moves": ref_moves,
    }


# ------------------------------------------------------------------------------
# Suite 4: Cross-Board Curriculum Transfer Arena (Table V)
# ------------------------------------------------------------------------------
def benchmark_curriculum_transfer(ckpt_info: dict, blocks: int, channels: int, quick: bool = False) -> list:
    trained = ckpt_info["trained_by_board"]
    fallback = ckpt_info.get("dry_run_fallback")
    num_games = 4 if quick else 20
    eval_sims_map = {13: 16, 19: 12} if quick else {13: 160, 19: 160}
    max_moves = 80 if quick else None

    results = []

    def _load_onto_board(entry: dict, target_board: int) -> torch.nn.Module:
        meta = og.load_checkpoint_auto(entry["path"])
        net = og.PolicyValueNet(board_size=target_board, num_channels=meta["channels"], num_res_blocks=meta["blocks"])
        net.load_state_dict(meta["state_dict"], strict=True)
        return _prep_net_for_device(net)

    # Zero-shot 9x9 -> 13x13 vs Tabula Rasa 13x13
    src_9 = trained.get(9) or fallback
    if src_9 is not None:
        is_smoke = (src_9 is fallback)
        tag = " [SMOKE/UNTRAINED]" if is_smoke else ""
        print(f"    -> Zero-Shot Transfer Arena on 13x13 ({src_9['filename']} vs Random Init, {num_games}g){tag}...", flush=True)
        torch.manual_seed(42)
        rand_13 = _prep_net_for_device(og.PolicyValueNet(board_size=13, num_channels=src_9["channels"], num_res_blocks=src_9["blocks"]))
        xfer_13 = _load_onto_board(src_9, 13)
        res = run_custom_arena(
            xfer_13, rand_13, board_size=13, num_games=num_games,
            cand_sims=eval_sims_map[13], ref_sims=eval_sims_map[13],
            max_moves_override=max_moves,
        )
        results.append({
            "matchup": f"Zero-Shot 9x9 -> 13x13 vs Random Init{tag}",
            "target_board": 13,
            "cand_source": f"{src_9['filename']} (iter={src_9['iteration']})",
            "ref_source": "Tabula Rasa (Random Init)",
            "eval_sims": eval_sims_map[13],
            "is_smoke_only": is_smoke,
            **res,
        })
        del rand_13, xfer_13
    else:
        results.append({"matchup": "Zero-Shot 9x9 -> 13x13 vs Random Init", "target_board": 13, "skipped": "No 9x9 checkpoint found"})

    # Zero-shot 13x13 (or 9x9) -> 19x19 vs Tabula Rasa 19x19
    src_13 = trained.get(13) or trained.get(9) or fallback
    if src_13 is not None:
        is_smoke = (src_13 is fallback)
        src_b = src_13["board_size"] if not is_smoke else 9
        tag = " [SMOKE/UNTRAINED]" if is_smoke else ""
        print(f"    -> Zero-Shot Transfer Arena on 19x19 ({src_13['filename']} vs Random Init, {num_games}g){tag}...", flush=True)
        torch.manual_seed(43)
        rand_19 = _prep_net_for_device(og.PolicyValueNet(board_size=19, num_channels=src_13["channels"], num_res_blocks=src_13["blocks"]))
        xfer_19 = _load_onto_board(src_13, 19)
        res = run_custom_arena(
            xfer_19, rand_19, board_size=19, num_games=num_games,
            cand_sims=eval_sims_map[19], ref_sims=eval_sims_map[19],
            max_moves_override=max_moves,
        )
        results.append({
            "matchup": f"Zero-Shot {src_b}x{src_b} -> 19x19 vs Random Init{tag}",
            "target_board": 19,
            "cand_source": f"{src_13['filename']} (iter={src_13['iteration']})",
            "ref_source": "Tabula Rasa (Random Init)",
            "eval_sims": eval_sims_map[19],
            "is_smoke_only": is_smoke,
            **res,
        })
        del rand_19, xfer_19
    else:
        results.append({"matchup": "Zero-Shot 13x13 -> 19x19 vs Random Init", "target_board": 19, "skipped": "No 13x13 or 9x9 checkpoint found"})

    # Trained Stage-2 vs Stage-1 (on 13x13) and Stage-3 vs Stage-2 (on 19x19) if trained checkpoints exist
    if 9 in trained and 13 in trained:
        print(f"    -> Stage Progression Arena on 13x13 (Trained 13x13 vs Zero-Shot 9x9, {num_games}g)...", flush=True)
        net_13 = _load_onto_board(trained[13], 13)
        net_9_on_13 = _load_onto_board(trained[9], 13)
        res = run_custom_arena(
            net_13, net_9_on_13, board_size=13, num_games=num_games,
            cand_sims=eval_sims_map[13], ref_sims=eval_sims_map[13],
            max_moves_override=max_moves,
        )
        results.append({
            "matchup": "Trained 13x13 vs Zero-Shot 9x9 (on 13x13)",
            "target_board": 13,
            "cand_source": f"{trained[13]['filename']} (iter={trained[13]['iteration']})",
            "ref_source": f"{trained[9]['filename']} (iter={trained[9]['iteration']})",
            "eval_sims": eval_sims_map[13],
            "is_smoke_only": False,
            **res,
        })
        del net_13, net_9_on_13

    if 13 in trained and 19 in trained:
        print(f"    -> Stage Progression Arena on 19x19 (Trained 19x19 vs Zero-Shot 13x13, {num_games}g)...", flush=True)
        net_19 = _load_onto_board(trained[19], 19)
        net_13_on_19 = _load_onto_board(trained[13], 19)
        res = run_custom_arena(
            net_19, net_13_on_19, board_size=19, num_games=num_games,
            cand_sims=eval_sims_map[19], ref_sims=eval_sims_map[19],
            max_moves_override=max_moves,
        )
        results.append({
            "matchup": "Trained 19x19 vs Zero-Shot 13x13 (on 19x19)",
            "target_board": 19,
            "cand_source": f"{trained[19]['filename']} (iter={trained[19]['iteration']})",
            "ref_source": f"{trained[13]['filename']} (iter={trained[13]['iteration']})",
            "eval_sims": eval_sims_map[19],
            "is_smoke_only": False,
            **res,
        })
        del net_19, net_13_on_19

    return results


# ------------------------------------------------------------------------------
# Suite 5: Symmetry Ensembling & Subtree Reuse Scaling Arena (Table VI)
# ------------------------------------------------------------------------------
def benchmark_symmetry_and_reuse(ckpt_info: dict, blocks: int, channels: int, quick: bool = False) -> list:
    trained = ckpt_info["trained_by_board"]
    fallback = ckpt_info.get("dry_run_fallback")

    # Prefer trained 9x9 checkpoint (fast Table VI evaluation) or highest trained stage, else fallback
    src = trained.get(9) or (max(trained.values(), key=lambda e: e["iteration"]) if trained else fallback)
    is_smoke = False
    if src is not None:
        meta = og.load_checkpoint_auto(src["path"])
        b_sz = int(src["board_size"]) if not (src is fallback) else 9
        net = og.PolicyValueNet(board_size=b_sz, num_channels=meta["channels"], num_res_blocks=meta["blocks"])
        net.load_state_dict(meta["state_dict"], strict=True)
        net_desc = f"{src['filename']} (iter={src['iteration']})"
        is_smoke = (src is fallback)
    else:
        b_sz = 9
        net = og.PolicyValueNet(board_size=b_sz, num_channels=channels, num_res_blocks=blocks)
        net_desc = "untrained_init"
        is_smoke = True

    net = _prep_net_for_device(net)
    num_games = 4 if quick else 100
    max_moves = 70 if quick else None
    tag = " [SMOKE/UNTRAINED]" if is_smoke else ""

    if quick:
        match_specs = [
            {"label": f"16 sims (8x Sym) vs 16 sims (1x View){tag}", "cand_sims": 16, "ref_sims": 16, "cand_sym": 8, "ref_sym": 1, "cand_reuse": True, "ref_reuse": True},
            {"label": f"32 sims (8x Sym) vs 32 sims (1x View){tag}", "cand_sims": 32, "ref_sims": 32, "cand_sym": 8, "ref_sym": 1, "cand_reuse": True, "ref_reuse": True},
            {"label": f"32 sims + Reuse vs 16 sims No Reuse{tag}", "cand_sims": 32, "ref_sims": 16, "cand_sym": 1, "ref_sym": 1, "cand_reuse": True, "ref_reuse": False},
        ]
    else:
        match_specs = [
            {"label": "100 sims (8x Sym Ensemble) vs 100 sims (1x Single View)", "cand_sims": 100, "ref_sims": 100, "cand_sym": 8, "ref_sym": 1, "cand_reuse": True, "ref_reuse": True},
            {"label": "400 sims (8x Sym Ensemble) vs 400 sims (1x Single View)", "cand_sims": 400, "ref_sims": 400, "cand_sym": 8, "ref_sym": 1, "cand_reuse": True, "ref_reuse": True},
            {"label": "600 sims + Subtree Reuse vs 200 sims No Reuse", "cand_sims": 600, "ref_sims": 200, "cand_sym": 1, "ref_sym": 1, "cand_reuse": True, "ref_reuse": False},
        ]

    rows = []
    for sp in match_specs:
        print(f"    -> Arena ({b_sz}x{b_sz}, {num_games}g): {sp['label']}...", flush=True)
        res = run_custom_arena(
            cand_net=net,
            ref_net=net,
            board_size=b_sz,
            num_games=num_games,
            cand_sims=sp["cand_sims"],
            ref_sims=sp["ref_sims"],
            cand_symmetries=sp["cand_sym"],
            ref_symmetries=sp["ref_sym"],
            cand_reuse=sp["cand_reuse"],
            ref_reuse=sp["ref_reuse"],
            max_moves_override=max_moves,
        )
        rows.append({
            "matchup": sp["label"],
            "board_size": b_sz,
            "checkpoint": net_desc,
            "is_smoke_only": is_smoke,
            **res,
        })

    del net
    return rows


# ------------------------------------------------------------------------------
# Suite 6: Training Log Parser (curriculum_console.log & logs_*.log)
# ------------------------------------------------------------------------------
def find_log_files(explicit_log: str = "") -> list:
    if explicit_log:
        return [explicit_log] if os.path.isfile(explicit_log) else []
    found = []
    if os.path.isfile("curriculum_console.log"):
        found.append("curriculum_console.log")
    try:
        log_candidates = sorted(
            [f for f in os.listdir(".") if f.startswith("logs_") and f.endswith(".log")],
            key=lambda p: os.path.getmtime(p),
        )
        found.extend(log_candidates)
    except OSError:
        pass
    return found


def parse_training_logs(log_files: list) -> dict:
    """
    Parses iteration-level telemetry from OceanGo `logs_*.log` and `curriculum_console.log`
    and aggregates summary metrics per board stage (9x9, 13x13, 19x19).
    """
    if not log_files:
        return {"files_parsed": [], "iterations": [], "stages": {}}

    iter_hdr_re = re.compile(r"====\s*Iteration\s+(\d+)/(\d+)\s+\(Board:\s*(\d+)x\d+.*?(?:Stage\s*Iter|StageIter)\s*[:=]\s*(\d+)\s*[,|]\s*LR:\s*([0-9.eE+-]+)")
    sp_log_re = re.compile(r"Self-Play\s*\((\d+)g\s+in\s+([0-9.]+)s,\s*([0-9.]+)g/s\):\s*\+(\d+)\s+states\s*\(Buf=(\d+)\),\s*AvgLen=([0-9.]+),\s*Resign=(\d+)/(\d+)\s*\(False=([0-9.]+)%\),\s*H=([0-9.]+)")
    sp_con_re = re.compile(r"Self-Play Telemetry:\s*\+([\d,]+)\s+unique states\s*\(Buffer:\s*([\d,]+)\)\s*\|\s*Avg Len:\s*([0-9.]+)\s*\|\s*Resigned:\s*(\d+)/(\d+)\s*\|\s*False-Resign:\s*([0-9.]+)%\s*\((\d+)/(\d+)\)\s*\|\s*Search Entropy:\s*([0-9.]+)")
    sp_bar_re = re.compile(r"Self-Play:\s*\[.*?\]\s*(\d+)/(\d+).*?\|\s*([0-9.]+)\s*games/s")
    tr_log_re = re.compile(r"Train\s*\((\d+)\s*steps\s+in\s+([0-9.]+)s\):\s*Loss=([0-9.]+)\s*\(P=([0-9.]+)\s+V=([0-9.]+)\s+O=([0-9.]+)\s+H=([0-9.]+)\)")
    tr_bar_re = re.compile(r"Train\s*\[.*?\]\s*(\d+)/(\d+).*?Loss:\s*([0-9.]+)\s*\(P:([0-9.]+)\s+V:([0-9.]+)\s+O:([0-9.]+)\s+H:([0-9.]+)\)\s*\|\s*([0-9.]+)\s*step/s")
    ar_log_re = re.compile(r"Arena:\s*(\d+)W-(\d+)L-(\d+)D\s*\(WR=([0-9.]+)%,\s*Elo=([0-9.]+)\)")
    ar_con_re = re.compile(r"Arena Result.*?:\s*(\d+)W-(\d+)L-(\d+)D\s*\(Win Rate:\s*([0-9.]+)%,\s*ΔElo:\s*([+-]?[0-9.]+)\s*->\s*Est\.\s*Elo:\s*([0-9.]+)\)")
    dur_re = re.compile(r"(?:Iteration\s+\d+\s+saved to.*?in|Duration:)\s*([0-9.]+)s")

    # Prefer detailed logs_*.log over sparse console logs when both cover the same (board, iter)
    records = {}
    parsed_files = []

    for lf in log_files:
        if not os.path.isfile(lf):
            continue
        parsed_files.append(lf)
        cur_rec = None
        try:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    m_it = iter_hdr_re.search(line)
                    if m_it:
                        it_num = int(m_it.group(1))
                        b_sz = int(m_it.group(3))
                        st_it = int(m_it.group(4))
                        lr = float(m_it.group(5))
                        key = (b_sz, it_num)
                        cur_rec = records.setdefault(key, {
                            "board_size": b_sz,
                            "iteration": it_num,
                            "stage_iter": st_it,
                            "lr": lr,
                        })
                        continue
                    if cur_rec is None:
                        continue

                    m_sp = sp_log_re.search(line)
                    if m_sp:
                        cur_rec.update({
                            "games": int(m_sp.group(1)),
                            "sp_elapsed_sec": float(m_sp.group(2)),
                            "games_per_sec": float(m_sp.group(3)),
                            "new_states": int(m_sp.group(4)),
                            "buffer_size": int(m_sp.group(5)),
                            "avg_len": float(m_sp.group(6)),
                            "resigned": int(m_sp.group(7)),
                            "total_games_sp": int(m_sp.group(8)),
                            "false_resign_pct": float(m_sp.group(9)),
                            "search_entropy": float(m_sp.group(10)),
                        })
                        continue

                    m_spc = sp_con_re.search(line)
                    if m_spc:
                        cur_rec.setdefault("new_states", int(m_spc.group(1).replace(",", "")))
                        cur_rec.setdefault("buffer_size", int(m_spc.group(2).replace(",", "")))
                        cur_rec.setdefault("avg_len", float(m_spc.group(3)))
                        cur_rec.setdefault("resigned", int(m_spc.group(4)))
                        cur_rec.setdefault("games", int(m_spc.group(5)))
                        cur_rec.setdefault("false_resign_pct", float(m_spc.group(6)))
                        cur_rec.setdefault("false_resign_checks", int(m_spc.group(8)))
                        cur_rec.setdefault("search_entropy", float(m_spc.group(9)))
                        continue

                    m_sb = sp_bar_re.search(line)
                    if m_sb:
                        cur_rec.setdefault("games_per_sec", float(m_sb.group(3)))
                        continue

                    m_tr = tr_log_re.search(line)
                    if m_tr:
                        cur_rec.update({
                            "train_steps": int(m_tr.group(1)),
                            "train_elapsed_sec": float(m_tr.group(2)),
                            "loss": float(m_tr.group(3)),
                            "policy_loss": float(m_tr.group(4)),
                            "value_loss": float(m_tr.group(5)),
                            "ownership_loss": float(m_tr.group(6)),
                            "pred_entropy": float(m_tr.group(7)),
                        })
                        continue

                    m_tb = tr_bar_re.search(line)
                    if m_tb:
                        cur_rec.setdefault("train_steps", int(m_tb.group(1)))
                        cur_rec.setdefault("loss", float(m_tb.group(3)))
                        cur_rec.setdefault("policy_loss", float(m_tb.group(4)))
                        cur_rec.setdefault("value_loss", float(m_tb.group(5)))
                        cur_rec.setdefault("ownership_loss", float(m_tb.group(6)))
                        cur_rec.setdefault("pred_entropy", float(m_tb.group(7)))
                        cur_rec["train_steps_per_sec"] = float(m_tb.group(8))
                        continue

                    m_ar = ar_log_re.search(line)
                    if m_ar:
                        cur_rec.update({
                            "arena_w": int(m_ar.group(1)),
                            "arena_l": int(m_ar.group(2)),
                            "arena_d": int(m_ar.group(3)),
                            "arena_wr": float(m_ar.group(4)),
                            "elo": float(m_ar.group(5)),
                        })
                        continue

                    m_ac = ar_con_re.search(line)
                    if m_ac:
                        cur_rec.setdefault("arena_w", int(m_ac.group(1)))
                        cur_rec.setdefault("arena_l", int(m_ac.group(2)))
                        cur_rec.setdefault("arena_d", int(m_ac.group(3)))
                        cur_rec.setdefault("arena_wr", float(m_ac.group(4)))
                        cur_rec.setdefault("elo", float(m_ac.group(6)))
                        continue

                    m_du = dur_re.search(line)
                    if m_du:
                        cur_rec["iter_duration_sec"] = float(m_du.group(1))
        except OSError:
            continue

    ordered = [records[k] for k in sorted(records.keys(), key=lambda x: (x[1], x[0]))]
    stages = {}
    for b_sz in (9, 13, 19):
        sub = [r for r in ordered if r["board_size"] == b_sz and "loss" in r]
        if not sub:
            sub = [r for r in ordered if r["board_size"] == b_sz]
        if not sub:
            continue
        completed = [r for r in sub if ("games" in r or "loss" in r)]
        if not completed:
            stages[str(b_sz)] = {"started_only": True, "iterations_started": len(sub)}
            continue

        def _avg(key):
            vals = [r[key] for r in completed if key in r]
            return (sum(vals) / len(vals)) if vals else None

        elos = [r["elo"] for r in completed if "elo" in r]
        total_dur_sec = sum(r.get("iter_duration_sec", r.get("sp_elapsed_sec", 0.0) + r.get("train_elapsed_sec", 0.0)) for r in completed)
        total_games = sum(r.get("games", 0) for r in completed)
        total_resigned = sum(r.get("resigned", 0) for r in completed)

        stages[str(b_sz)] = {
            "started_only": False,
            "iterations_completed": len(completed),
            "total_games": total_games,
            "total_states": sum(r.get("new_states", 0) for r in completed),
            "total_wall_hours": total_dur_sec / 3600.0,
            "avg_games_per_sec": _avg("games_per_sec"),
            "avg_game_len": _avg("avg_len"),
            "resign_rate_pct": (100.0 * total_resigned / total_games) if total_games > 0 else None,
            "avg_false_resign_pct": _avg("false_resign_pct"),
            "final_loss": completed[-1].get("loss"),
            "final_policy_loss": completed[-1].get("policy_loss"),
            "final_value_loss": completed[-1].get("value_loss"),
            "final_ownership_loss": completed[-1].get("ownership_loss"),
            "final_search_entropy": completed[-1].get("search_entropy"),
            "start_elo": elos[0] if elos else None,
            "final_elo": elos[-1] if elos else None,
            "peak_elo": max(elos) if elos else None,
        }

    return {"files_parsed": parsed_files, "iterations": ordered, "stages": stages}


# ------------------------------------------------------------------------------
# Report Formatting (Markdown Tables Ready for IEEE Draft Section VI)
# ------------------------------------------------------------------------------
def build_markdown_report(results: dict) -> str:
    lines = []
    meta = results["metadata"]
    lines.append(f"# OceanGo V2 IEEE Paper Benchmark Report (`{meta['machine_slug']}`)")
    lines.append(f"- **Generated At**: {meta['timestamp']}")
    lines.append(f"- **Quick Mode**: `{meta['quick']}`\n")

    if "hw" in results:
        hw = results["hw"]
        lines.append("## Table II Row: Hardware Profile")
        lines.append("| Platform | OS | CPU / Workers | Accelerator & Precision | RAM | Backbone (Params) |")
        lines.append("|:---|:---|:---|:---|:---:|:---|")
        lines.append(
            f"| **{hw['cpu_brand']}** | {hw['os']} | {hw['cpu_logical_threads']}T -> {hw['workers_used']}W | "
            f"{hw['device_name']} (`{hw['precision']}` + `{hw['memory_layout']}`) | {hw['system_ram_gb']:.1f} GB | "
            f"`{hw['active_backbone']}` ({hw['active_params']:,}) |"
        )
        lines.append("\nParameter counts verified: `15x192` = "
                     f"`{hw['param_counts'].get('15x192', 0):,}`, `18x256` = `{hw['param_counts'].get('18x256', 0):,}`.\n")

    if "engine" in results:
        lines.append("## Table III: Go Rules Engine State-Traversal & Memory Compression")
        lines.append("| Board Size | Legacy V1 Engine (`deepcopy` + DFS) | OceanGo V2 Engine (Zobrist + Bitmasks) | Speedup | Raw `float32[18,N,N]` | V2 `uint16[N,N]` | Compression | 500k Full Buffer RAM |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in results["engine"]:
            n = r["board_size"]
            lines.append(
                f"| **`{n}x{n}`** | `{r['v1_ply_per_sec']:,.1f}` ply/s | `{r['v2_ply_per_sec']:,.1f}` ply/s | "
                f"**{r['speedup']:.1f}x** | `{r['raw_float32_bytes']:,}` B | `{r['packed_uint16_bytes']:,}` B | "
                f"**{r['state_compression']:.1f}x** | `{r['replay_500k_gb']:.2f} GB` |"
            )
        lines.append("")

    if "ipc" in results:
        lines.append("## Table IV: Parallel Self-Play IPC & SGD Training Throughput")
        lines.append("| Board Size | Model Source | Pipe-Pickled Queue (games/s) | Zero-Pickle `SharedMemory` + Playout Caps (games/s) | Speedup | Measured GPU Evals/s | Effective Full-Budget Sims/s | SGD Train Speed (steps/s) |")
        lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in results["ipc"]:
            n = r["board_size"]
            lines.append(
                f"| **`{n}x{n}`** (@`{r['n_sims_full']}` sims, `{r['workers']}W`) | `{r['model_source']}` | "
                f"`{r['pipe_games_per_sec']:.3f}` | **`{r['shm_games_per_sec']:.3f}`** | **{r['games_speedup']:.2f}x** | "
                f"`{r['shm_nn_evals_per_sec']:,.1f}` | `{r['shm_effective_sims_per_sec']:,.1f}` | "
                f"`{r['sgd_steps_per_sec']:.2f}` (`B={r['sgd_batch_size']}`, `{r['sgd_samples_per_sec']:,.0f}` samp/s) |"
            )
        lines.append("")

    if "transfer" in results:
        lines.append("## Table V: Cross-Board Curriculum Transfer Arena")
        lines.append("| Matchup | Target Board | Candidate vs Reference (`W-L-D`) | Candidate Win Rate | Estimated $\\Delta$Elo | Avg Latency (Cand / Ref) |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for r in results["transfer"]:
            if "skipped" in r:
                lines.append(f"| {r['matchup']} | `{r['target_board']}x{r['target_board']}` | *SKIPPED ({r['skipped']})* | — | — | — |")
            else:
                lines.append(
                    f"| **{r['matchup']}** (`{r['eval_sims']}` sims) | `{r['target_board']}x{r['target_board']}` | "
                    f"`{r['wins']}W-{r['losses']}L-{r['draws']}D` | **{r['win_rate']*100:.1f}%** | `{r['elo_diff']:+.0f}` | "
                    f"`{r['cand_avg_ms_per_move']:.1f}` / `{r['ref_avg_ms_per_move']:.1f}` ms/move |"
                )
        lines.append("")

    if "arena" in results:
        lines.append("## Table VI: Symmetry Ensembling & Subtree Reuse Arena")
        lines.append("| Matchup Configuration | Board | Record (`W-L-D`) | Candidate Win Rate | Estimated $\\Delta$Elo | Avg Latency (Cand / Ref) |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for r in results["arena"]:
            lines.append(
                f"| **{r['matchup']}** | `{r['board_size']}x{r['board_size']}` | "
                f"`{r['wins']}W-{r['losses']}L-{r['draws']}D` | **{r['win_rate']*100:.1f}%** | `{r['elo_diff']:+.0f}` | "
                f"`{r['cand_avg_ms_per_move']:.1f}` / `{r['ref_avg_ms_per_move']:.1f}` ms/move |"
            )
        lines.append("")

    if "logs" in results:
        lg = results["logs"]
        lines.append("## Section VI-C: Parsed Training Log Telemetry")
        if not lg["stages"]:
            lines.append("*No completed training iterations found in log files yet.*\n")
        else:
            lines.append("| Stage | Completed Iters | Total Games | Total Hours | Avg `games/s` | Avg Len | Resign % | False-Resign % | Final Loss (P / V / O) | Final / Peak Elo |")
            lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|")
            for b_key in ("9", "13", "19"):
                if b_key not in lg["stages"]:
                    continue
                st = lg["stages"][b_key]
                if st.get("started_only"):
                    lines.append(f"| **`{b_key}x{b_key}`** | 0 completed ({st['iterations_started']} started) | — | — | — | — | — | — | — | — |")
                    continue
                gps = f"`{st['avg_games_per_sec']:.2f}`" if st["avg_games_per_sec"] is not None else "—"
                alen = f"`{st['avg_game_len']:.1f}`" if st["avg_game_len"] is not None else "—"
                rrate = f"`{st['resign_rate_pct']:.1f}%`" if st["resign_rate_pct"] is not None else "—"
                frate = f"`{st['avg_false_resign_pct']:.1f}%`" if st["avg_false_resign_pct"] is not None else "—"
                losses = (
                    f"`{st['final_loss']:.3f}` (`{st['final_policy_loss']:.3f}` / `{st['final_value_loss']:.3f}` / `{st['final_ownership_loss']:.3f}`)"
                    if st["final_loss"] is not None else "—"
                )
                elo_s = f"`{st['final_elo']:.0f}` / `{st['peak_elo']:.0f}`" if st["final_elo"] is not None else "—"
                lines.append(
                    f"| **`{b_key}x{b_key}`** | {st['iterations_completed']} | {st['total_games']:,} | `{st['total_wall_hours']:.2f}h` | "
                    f"{gps} | {alen} | {rrate} | {frate} | {losses} | {elo_s} |"
                )
            lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------------------
# Main CLI Entrypoint
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OceanGo V2 — IEEE Paper Benchmark & Log Extractor")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--log-file", type=str, default="", help="Specific training log file (default: auto-detect)")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "hw", "engine", "ipc", "transfer", "arena", "logs"],
        help="Benchmark suite to run (default: all)",
    )
    parser.add_argument("--quick", action="store_true", help="Fast smoke verification mode (1-2 min)")
    parser.add_argument("--blocks", type=int, default=0, help="Override ResNet blocks (default: auto-detect)")
    parser.add_argument("--channels", type=int, default=0, help="Override ResNet channels (default: auto-detect)")
    parser.add_argument("--workers", type=int, default=0, help=f"CPU worker count (default: auto={og.PARALLEL_THREADS})")
    parser.add_argument("--out-dir", type=str, default="paper", help="Directory for Markdown/JSON exports (default: paper)")
    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else og.PARALLEL_THREADS
    log_files = find_log_files(args.log_file)
    ckpt_info = discover_checkpoints(args.save_dir)

    log_blocks, log_channels = detect_arch_from_logs(log_files)
    blocks = args.blocks if args.blocks > 0 else (ckpt_info["detected_blocks"] or log_blocks or og.RESNET_BLOCKS)
    channels = args.channels if args.channels > 0 else (ckpt_info["detected_channels"] or log_channels or og.RESNET_CHANNELS)

    suites = ["hw", "engine", "ipc", "transfer", "arena", "logs"] if args.suite == "all" else [args.suite]

    print("=" * 72, flush=True)
    print(f"OceanGo V2 IEEE Paper Benchmark Runner (quick={args.quick})", flush=True)
    print(f"  Backbone  : {blocks} blocks x {channels} channels | Workers: {workers}", flush=True)
    print(f"  Checkpoints ({args.save_dir}): trained stages = {sorted(ckpt_info['trained_by_board'].keys())}", flush=True)
    print(f"  Logs Found: {log_files}", flush=True)
    print("=" * 72, flush=True)

    results = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "quick": bool(args.quick),
            "save_dir": args.save_dir,
            "blocks": blocks,
            "channels": channels,
            "workers": workers,
        }
    }

    hw_prof = benchmark_hardware_profile(blocks=blocks, channels=channels, workers=workers)
    results["metadata"]["machine_slug"] = hw_prof["machine_slug"]

    mp_ctx = mp.get_context("spawn")

    if "hw" in suites:
        print("\n[Suite 1/6] Profiling hardware and parameter counts...", flush=True)
        results["hw"] = hw_prof
        print(f"  -> {hw_prof['cpu_brand']} | {hw_prof['device_name']} | {hw_prof['active_backbone']} ({hw_prof['active_params']:,} params)", flush=True)

    if "engine" in suites:
        print("\n[Suite 2/6] Running V1 vs V2 Go rules engine & replay memory benchmark...", flush=True)
        results["engine"] = benchmark_engine_v1_vs_v2(quick=args.quick)
        for r in results["engine"]:
            print(f"  -> {r['board_size']:2d}x{r['board_size']:<2d}: V1={r['v1_ply_per_sec']:8.1f} ply/s | V2={r['v2_ply_per_sec']:9.1f} ply/s | Speedup={r['speedup']:.1f}x", flush=True)

    if "ipc" in suites:
        print("\n[Suite 3/6] Running Pipe-Pickled Queue vs SharedMemory + Playout Caps & SGD benchmark...", flush=True)
        results["ipc"] = benchmark_ipc_and_training(ckpt_info, blocks=blocks, channels=channels, workers=workers, mp_ctx=mp_ctx, quick=args.quick)

    if "transfer" in suites:
        print("\n[Suite 4/6] Running Cross-Board Curriculum Transfer Arena...", flush=True)
        results["transfer"] = benchmark_curriculum_transfer(ckpt_info, blocks=blocks, channels=channels, quick=args.quick)

    if "arena" in suites:
        print("\n[Suite 5/6] Running Symmetry Ensembling & Subtree Reuse Arena...", flush=True)
        results["arena"] = benchmark_symmetry_and_reuse(ckpt_info, blocks=blocks, channels=channels, quick=args.quick)

    if "logs" in suites:
        print("\n[Suite 6/6] Parsing training logs...", flush=True)
        results["logs"] = parse_training_logs(log_files)
        print(f"  -> Parsed {len(results['logs']['iterations'])} iteration records across {len(results['logs']['files_parsed'])} log files.", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    slug = hw_prof["machine_slug"]
    json_path = os.path.join(args.out_dir, f"benchmark_results_{slug}.json")
    md_path = os.path.join(args.out_dir, f"benchmark_report_{slug}.md")

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)

    report_md = build_markdown_report(results)
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write(report_md)

    print("\n" + "=" * 72, flush=True)
    print(report_md, flush=True)
    print("=" * 72, flush=True)
    print(f"Saved JSON results : {json_path}", flush=True)
    print(f"Saved Markdown report: {md_path}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
