#!/usr/bin/env python3
# OceanGo - 9×9 (default) Go with GPU training & CPU self-play
# Features: progress bars, AMP, simple-Ko, suicide ban, GUI.
# --------------------------------------------------------------------------
import os, random, math, logging, warnings, multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

# ────────────── silence pygame banner ──────────────
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning,
                        module="pygame.pkgdata")
pygame = None        # lazy import inside GUI

# ───────────────────────── CONFIG ─────────────────────────
@dataclass
class Config:
    # ── 1. Board & GUI ────────────────────────────────────────
    board_size: int = 9        # Size of the Go board (NxN).
                               # ↑ : tensors grow ∝ N², MCTS gets slower,
                               #     GPU VRAM & CPU time up sharply.
                               # Resource  → CPU-time, GPU-VRAM.

    grid: int = 50             # Pixel width of one cell in the GUI.
                               # Only affects rendering – no perf impact.

    # ── 2. Self-play schedule (how much data you generate) ────
    n_iter: int = 15           # Outer training loops per run.
                               # ↑ : linear CPU time, more data in buffer.

    games_per_iter: int = 30   # Parallel self-play games each loop.
                               # ↑ : more CPU threads busy, more RAM used
                               #     to store new positions.

    # ── 3. MCTS search depth ─────────────────────────────────
    mcts_sims: int = 1000      # Simulations per move in MCTS.
                               # ↑ : CPU time per move rises linearly;
                               #     search stronger, GUI latency higher.

    # ── 4. Replay buffer (lives in **system RAM**) ────────────
    replay_cap: int = 10_000_000   # Maximum positions kept.
                                   # Each 9×9 record ≈0.5 KB ➜ 10 M ≈ 5 GB.
                                   # ↑ : needs more RAM, gives better
                                   #     training diversity.

    # ── 5. SGD / GPU training parameters ─────────────────────
    train_batch: int = 256     # Mini-batch size per optimizer step.
                               # ↑ : GPU VRAM use rises ~linearly;
                               #     gradients smoother. 8 GB RTX 3070
                               #     tops out ≈384 (AMP on).

    train_epochs: int = 5      # Passes over the sampled batch each loop.
                               # ↑ : more GPU compute time, no extra RAM.

    lr: float = 1e-4           # Adam learning rate.
                               # ↑ : faster learning but risk of divergence.

    # ── 6. Hardware toggles ───────────────────────────────────
    amp: bool = True           # Automatic Mixed Precision on GPU.
                               # Off ➜ +VRAM, -speed.

    num_workers: int = 24      # CPU processes that run self-play.
                               # ↑ : more cores used, faster data gen;
                               #     too high ➜ OS overhead / context-switch.

CFG = Config()
SCREEN = (CFG.board_size + 1) * CFG.grid
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── LOGGING ─────────────────────────
root = Path(__file__).resolve().parent
logging.basicConfig(
    filename=root / "ocean_go.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info("\n\n───────── New session ─────────")

# ==========================================================
#                   POLICY-VALUE NETWORK
# ==========================================================
class PolicyValueNet(nn.Module):
    def __init__(self, bs: int = CFG.board_size):
        super().__init__()
        self.bs = bs
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        # policy head
        self.p_conv = nn.Conv2d(64, 2, 1)
        self.p_fc = nn.Linear(2 * bs * bs, bs * bs + 1)
        # value head
        self.v_conv = nn.Conv2d(64, 1, 1)
        self.v_fc1 = nn.Linear(bs * bs, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        p = F.relu(self.p_conv(x))
        p = self.p_fc(p.view(p.size(0), -1))
        v = F.relu(self.v_conv(x))
        v = F.relu(self.v_fc1(v.view(v.size(0), -1)))
        v = torch.tanh(self.v_fc2(v))
        return p, v

# ==========================================================
#                     GO GAME LOGIC
# ==========================================================
def softmax_np(x):
    e = np.exp(x - x.max())
    return e / e.sum()

class GoGame:
    def __init__(self, bs: int = CFG.board_size):
        self.bs = bs
        self.board = np.zeros((bs, bs), np.int8)
        self.current_player = 1
        self.history: List[np.ndarray] = []   # snapshots for Ko
        self.ko_point: Tuple[int, int] | None = None

    # -------------- internal helpers ----------------
    def copy(self):
        g = GoGame(self.bs)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.history = [b.copy() for b in self.history]
        g.ko_point = self.ko_point
        return g

    def switch(self):
        self.current_player = 3 - self.current_player

    def is_valid(self, x, y):
        return 0 <= x < self.bs and 0 <= y < self.bs and self.board[x, y] == 0

    def _has_liberty(self, start, temp_board):
        from collections import deque
        color = temp_board[start]
        seen, dq = set(), deque([start])
        while dq:
            cx, cy = dq.pop()
            if (cx, cy) in seen:
                continue
            seen.add((cx, cy))
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.bs and 0 <= ny < self.bs:
                    if temp_board[nx, ny] == 0:
                        return True
                    if temp_board[nx, ny] == color and (nx, ny) not in seen:
                        dq.append((nx, ny))
        return False

    def is_suicide(self, x, y):
        if not self.is_valid(x, y): return False
        tmp = self.board.copy()
        tmp[x, y] = self.current_player
        self._remove_captured(3 - self.current_player, tmp)
        return not self._has_liberty((x, y), tmp)

    # -------------- captures & eyes ------------------
    def _find_group(self, start, board):
        from collections import deque
        color = board[start]
        group, dq, lib = set(), deque([start]), set()
        while dq:
            cx, cy = dq.pop()
            if (cx, cy) in group: continue
            group.add((cx, cy))
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.bs and 0 <= ny < self.bs:
                    if board[nx, ny] == 0:
                        lib.add((nx, ny))
                    elif board[nx, ny] == color and (nx, ny) not in group:
                        dq.append((nx, ny))
        return group, lib

    def _remove_captured(self, color, board=None):
        if board is None:
            board = self.board
        captured = []
        visited = set()
        for i in range(self.bs):
            for j in range(self.bs):
                if board[i, j] == color and (i, j) not in visited:
                    grp, lib = self._find_group((i, j), board)
                    visited |= grp
                    if not lib:
                        captured.extend(grp)
        for x, y in captured:
            board[x, y] = 0

    # -------------- move with simple-Ko check ----------
    def make_move(self, x, y):
        if not self.is_valid(x, y): return False
        if self.ko_point == (x, y): return False

        prev_board = self.board.copy()
        self.board[x, y] = self.current_player
        self._remove_captured(3 - self.current_player)

        # suicide?
        if not self._has_liberty((x, y), self.board):
            self.board[:] = prev_board
            return False

        # simple-Ko: board identical to position 2 plies ago?
        if self.history and np.array_equal(self.board, self.history[-1]):
            self.board[:] = prev_board
            return False

        # update Ko point (single-stone Ko)
        self.ko_point = None
        diff = np.where(prev_board != self.board)
        if diff[0].size == 1:  # one stone difference
            cx, cy = diff[0][0], diff[1][0]
            if prev_board[cx, cy] != 0 and self.board[cx, cy] == 0:
                self.ko_point = (cx, cy)

        self.history.append(prev_board)       # keep last state
        self.switch()
        return True

    # ---------- convenience for GUI / debug ------------
    def group_stats(self, x, y):
        """Return (#liberties, is_eye) for group containing (x,y)."""
        if self.board[x, y] == 0:
            return 0, False
        grp, lib = self._find_group((x, y), self.board)
        # eye heuristic: every liberty is fully surrounded by same color
        col = self.board[x, y]
        eye = all(all(self.board[nx + dx, ny + dy] == col
                       for dx, dy in ((1,0),(-1,0),(0,1),(0,-1))
                       if 0 <= nx + dx < self.bs and 0 <= ny + dy < self.bs)
                  for nx, ny in lib)
        return len(lib), eye

    # ---------- external interface ----------
    def get_legal_moves(self):
        return [(x, y) for x in range(self.bs) for y in range(self.bs)
                if self.board[x, y] == 0 and not self.is_suicide(x, y)
                and (x, y) != self.ko_point]

    def game_over(self):
        return not self.get_legal_moves()

    def winner(self):
        s1 = np.count_nonzero(self.board == 1)
        s2 = np.count_nonzero(self.board == 2)
        return 0 if s1 == s2 else (1 if s1 > s2 else 2)

    def state_tensor(self):
        p, o = self.current_player, 3 - self.current_player
        return torch.from_numpy(np.stack([(self.board == p).astype(np.float32),
                                          (self.board == o).astype(np.float32)], 0))

# ==========================================================
#                     MCTS  (NN-guided)
# ==========================================================
class Node:
    __slots__ = ("g", "p", "mv", "ch", "unexp", "N", "W", "Q", "P", "val")
    def __init__(self, g, parent=None, mv=None):
        self.g, self.p, self.mv = g, parent, mv
        self.ch: Dict[Tuple[int, int],"Node"] = {}
        self.unexp = g.get_legal_moves()
        self.N = self.W = self.Q = 0.0
        self.P = 0.0
        self.val = 0.0

def select(node: Node):
    while True:
        if node.unexp or not node.ch:
            return node
        node = max(node.ch.values(),
                   key=lambda c: c.Q + 1.0 * c.P * math.sqrt(node.N) / (1 + c.N))

@torch.no_grad()
def nn_eval(batch: List[Node], net: PolicyValueNet):
    dev = next(net.parameters()).device
    x = torch.stack([n.g.state_tensor() for n in batch]).to(dev)
    logit, v = net(x)
    logit, v = logit.cpu().numpy(), v.squeeze(1).cpu().numpy()
    for n, lvec, val in zip(batch, logit, v):
        n.val = float(val)
        if not n.unexp: continue
        idx = [m[0] * CFG.board_size + m[1] for m in n.unexp]
        pr = softmax_np(lvec[:CFG.board_size**2][idx])
        for m, p in zip(n.unexp, pr):
            ch = Node(n.g.copy(), n, m); ch.P = p
            n.ch[m] = ch
        n.unexp.clear()

def backup(n: Node, value: float):
    while n:
        n.N += 1
        n.W += value
        n.Q = n.W / n.N
        value = -value
        n = n.p

def mcts(game: GoGame, net: PolicyValueNet, sims=CFG.mcts_sims, temp=1.0):
    root = Node(game.copy()); nn_eval([root], net)
    for _ in range(sims):
        leaf = select(root); nn_eval([leaf], net); backup(leaf, leaf.val)
    moves, visits = zip(*((m, ch.N) for m, ch in root.ch.items()))
    v = np.array(visits, np.float32)
    if temp < 1e-3:
        p = np.zeros_like(v); p[np.argmax(v)] = 1.0
    else:
        w = v ** (1 / temp); p = w / w.sum()
    return dict(zip(moves, p))

# ==========================================================
#               SELF-PLAY WORKER  (CPU-only)
# ==========================================================
def self_play_worker(args):
    seed, model_path = args
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    net = PolicyValueNet(CFG.board_size).cpu()
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    game, records = GoGame(CFG.board_size), []
    while not game.game_over():
        pi = mcts(game, net, CFG.mcts_sims, 1.0)
        vec = np.zeros(CFG.board_size**2, np.float32)
        for mv, p in pi.items():
            vec[mv[0] * CFG.board_size + mv[1]] = p
        records.append((game.board.copy(), game.current_player, vec))
        mv = random.choices(list(pi.keys()), weights=list(pi.values()))[0]
        game.make_move(*mv)

    w = game.winner()
    return [(b, p, v, 0 if w == 0 else (1 if p == w else -1)) for b, p, v in records]

# ==========================================================
#                     REPLAY BUFFER
# ==========================================================
class Replay:
    def __init__(self, cap): self.data, self.cap = [], cap
    def add(self, new): self.data.extend(new); self.data = self.data[-self.cap:]
    def sample(self, n): return self.data if n >= len(self.data) else random.sample(self.data, n)
RB = Replay(CFG.replay_cap)

# ==========================================================
#                       TRAINING
# ==========================================================
def planes(board, player):
    return torch.from_numpy(np.stack([(board == player).astype(np.float32),
                                      (board == 3 - player).astype(np.float32)], 0))

def batch_tensor(batch):
    b, pl, pi, z = zip(*batch)
    X = torch.stack([planes(bb, pp) for bb, pp in zip(b, pl)]).to(device_gpu)
    return X, torch.tensor(pi, device=device_gpu), torch.tensor(z, dtype=torch.float32,
                                                                device=device_gpu).unsqueeze(1)

def train(net: PolicyValueNet, batch, it_idx):
    opt = optim.Adam(net.parameters(), lr=CFG.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.amp)
    net.train()
    X, pi_t, z_t = batch_tensor(batch)
    for ep in range(CFG.train_epochs):
        idx = np.arange(len(batch)); np.random.shuffle(idx)
        bar = tqdm(range(0, len(batch), CFG.train_batch),
                   desc=f"Iter {it_idx} | Epoch {ep+1}", leave=False)
        avg = 0.0                            
        step = 0         
        for s in bar:
            sam = idx[s:s+CFG.train_batch]
            inp, tgt_pi, tgt_z = X[sam], pi_t[sam], z_t[sam]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=CFG.amp):
                out_p, out_v = net(inp)
                out_p = out_p[:, :CFG.board_size**2]
                loss_p = -(tgt_pi * F.log_softmax(out_p, 1)).sum(1).mean()
                loss_v = F.mse_loss(out_v, tgt_z)
                loss = loss_p + loss_v
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            step += 1; avg += (loss.item() - avg) / step
            bar.set_postfix(loss=f"{loss.item():.4f}")
        logging.info("Iter %d | Epoch %d | loss %.4f", it_idx, ep+1, loss.item())
    net.eval()

# ==========================================================
#                         GUI
# ==========================================================
def run_gui(net: PolicyValueNet):
    global pygame
    import pygame as _pg; pygame = _pg

    pygame.init()
    screen = pygame.display.set_mode((SCREEN, SCREEN))
    pygame.display.set_caption("OceanGo")

    def draw(g: GoGame):
        screen.fill((240, 207, 155))
        for i in range(1, CFG.board_size + 1):
            pygame.draw.line(screen, (0,0,0),
                             (i * CFG.grid, CFG.grid),
                             (i * CFG.grid, SCREEN - CFG.grid), 2)
            pygame.draw.line(screen, (0,0,0),
                             (CFG.grid, i * CFG.grid),
                             (SCREEN - CFG.grid, i * CFG.grid), 2)
        for x in range(CFG.board_size):
            for y in range(CFG.board_size):
                s = g.board[x, y]
                if s:
                    col = (0,0,0) if s == 1 else (240,240,240)
                    pygame.draw.circle(screen, col,
                                       ((y + 1) * CFG.grid, (x + 1) * CFG.grid),
                                       CFG.grid // 2 - 4)
        pygame.display.flip()

    game, clock = GoGame(CFG.board_size), pygame.time.Clock()
    while True:
        if game.game_over():
            print("Winner:", game.winner())
            pygame.time.wait(3000); return
        draw(game); clock.tick(30)

        if game.current_player == 1:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: return
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = ev.pos
                    gx, gy = round(my / CFG.grid - 1), round(mx / CFG.grid - 1)
                    game.make_move(gx, gy)
        else:
            probs = mcts(game, net, CFG.mcts_sims, temp=0.1)
            if not probs: print("AI passes."); return
            best = max(probs.items(), key=lambda kv: kv[1])[0]
            game.make_move(*best)

# ==========================================================
#                           MAIN
# ==========================================================
def main():
    print("OceanGo - CUDA:", torch.cuda.is_available(), "- AMP:", CFG.amp)
    torch.backends.cudnn.benchmark = True
    net = PolicyValueNet().to(device_gpu)

    ckpt = root / "policy_value_net.pth"
    do_train = True
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device_gpu)
        cur = net.state_dict(); cur.update(state); net.load_state_dict(cur, strict=False)
        print("Found model, loaded. ")
        try:
            if input("Train further (T) or play GUI (G)? [T/G]: ").strip().lower() == "g":
                do_train = False
        except EOFError:
            pass
    else:
        print("No checkpoint - starting fresh.")

    if not do_train:
        run_gui(net); return

    tmp = root / "_tmp_cpu.pth"; torch.save(net.cpu().state_dict(), tmp); net.to(device_gpu)
    outer = tqdm(range(1, CFG.n_iter + 1), desc="Iterations")
    for it in outer:
        seeds = [random.randrange(2**32) for _ in range(CFG.games_per_iter)]
        with mp.Pool(CFG.num_workers) as pool:
            bar = tqdm(total=CFG.games_per_iter, desc=f"Iter {it} self-play", leave=False)
            moves_total = 0
            for rec in pool.imap_unordered(self_play_worker, [(s, tmp.as_posix()) for s in seeds]):
                RB.add(rec); bar.update()
                moves_total += len(rec)             # <<< count moves
                bar.set_postfix(buffer=len(RB.data), moves=moves_total)  # <<<
            bar.close()
        train(net, RB.sample(1024), it)
        torch.save(net.state_dict(), ckpt)
        torch.save(net.cpu().state_dict(), tmp)
        net.to(device_gpu)
        outer.set_postfix(buffer=len(RB.data))  

    run_gui(net)

# ───────────────────────── entry ─────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
