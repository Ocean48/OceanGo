#!/usr/bin/env python3
# OceanGo – 13×13 Go with GPU training & CPU self-play
# Progress-bar edition (tqdm) – fixed logging.basicConfig bug
# ---------------------------------------------------------------------------
import os, sys, math, random, logging, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp
from tqdm.auto import tqdm

# ─────────────────── silence pygame banner ───────────────────
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module="pygame.pkgdata")
pygame = None  # lazy import later

# ───────────────────────────── CONFIG ─────────────────────────
@dataclass
class Config:
    board_size: int = 13
    grid: int = 50
    n_iter: int = 9
    games_per_iter: int = 9
    mcts_sims: int = 500
    replay_cap: int = 200_000
    train_batch: int = 256
    train_epochs: int = 5
    lr: float = 1e-3
    amp: bool = True
    num_workers: int = 24

CFG = Config()
SCREEN = (CFG.board_size + 1) * CFG.grid
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────── LOGGING ──────────────────────────
root = Path(__file__).resolve().parent
logging.basicConfig(                       # ← fixed
    filename=root / "ocean_go.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info("\n\n───────── New session ─────────")

# =============================================================#
#                     POLICY–VALUE NET                          #
# =============================================================#
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
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = F.relu(self.conv3(x))
        p = F.relu(self.p_conv(x)); p = self.p_fc(p.view(p.size(0), -1))
        v = F.relu(self.v_conv(x)); v = F.relu(self.v_fc1(v.view(v.size(0), -1)))
        v = torch.tanh(self.v_fc2(v))
        return p, v

# =============================================================#
#                       GO GAME LOGIC                           #
# =============================================================#
class GoGame:
    def __init__(self, bs: int = CFG.board_size):
        self.bs = bs
        self.board = np.zeros((bs, bs), np.int8)
        self.current_player = 1

    def copy(self):
        g = GoGame(self.bs)
        g.board = self.board.copy()
        g.current_player = self.current_player
        return g

    def switch(self): self.current_player = 3 - self.current_player
    def is_valid(self, x, y): return 0 <= x < self.bs and 0 <= y < self.bs and self.board[x, y] == 0

    def _has_liberty(self, start, temp):
        from collections import deque
        color = temp[start]
        seen, dq = set(), deque([start])
        while dq:
            cx, cy = dq.pop()
            if (cx, cy) in seen: continue
            seen.add((cx, cy))
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < self.bs and 0 <= ny < self.bs:
                    if temp[nx, ny] == 0: return True
                    if temp[nx, ny] == color and (nx, ny) not in seen:
                        dq.append((nx, ny))
        return False

    def is_suicide(self, x, y):
        if not self.is_valid(x, y): return False
        tmp = self.board.copy(); tmp[x, y] = self.current_player
        return not self._has_liberty((x, y), tmp)

    def get_legal_moves(self):
        return [(x, y) for x in range(self.bs) for y in range(self.bs)
                if self.board[x, y] == 0 and not self.is_suicide(x, y)]

    def _find_group(self, start):
        from collections import deque
        color = self.board[start]
        group, dq, lib = set(), deque([start]), False
        while dq:
            cx, cy = dq.pop()
            if (cx, cy) in group: continue
            group.add((cx, cy))
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < self.bs and 0 <= ny < self.bs:
                    if self.board[nx, ny] == 0: lib = True
                    elif self.board[nx, ny] == color and (nx, ny) not in group:
                        dq.append((nx, ny))
        return group, lib

    def _remove_captured(self, color):
        captured, seen = [], set()
        for i in range(self.bs):
            for j in range(self.bs):
                if self.board[i, j] == color and (i, j) not in seen:
                    grp, lib = self._find_group((i, j)); seen.update(grp)
                    if not lib: captured.extend(grp)
        for x, y in captured: self.board[x, y] = 0

    def make_move(self, x, y):
        if not self.is_valid(x, y) or self.is_suicide(x, y): return False
        self.board[x, y] = self.current_player
        self._remove_captured(3 - self.current_player)
        self.switch(); return True

    def game_over(self): return not self.get_legal_moves()
    def winner(self):
        s1 = np.count_nonzero(self.board == 1)
        s2 = np.count_nonzero(self.board == 2)
        return 0 if s1 == s2 else (1 if s1 > s2 else 2)

    def state_tensor(self):
        p, o = self.current_player, 3 - self.current_player
        return torch.from_numpy(np.stack([(self.board == p).astype(np.float32),
                                          (self.board == o).astype(np.float32)], 0))

# =============================================================#
#                          MCTS                                 #
# =============================================================#
def softmax_np(x): e = np.exp(x - x.max()); return e / e.sum()

class Node:
    __slots__ = ("g", "p", "mv", "ch", "unexp", "N", "W", "Q", "P", "val")
    def __init__(self, g, parent=None, mv=None):
        self.g, self.p, self.mv = g, parent, mv
        self.ch = {}; self.unexp = g.get_legal_moves()
        self.N = self.W = self.Q = 0.0; self.P = 0.0; self.val = 0.0

def select(n: Node):
    while True:
        if n.unexp or not n.ch: return n
        n = max(n.ch.values(),
                key=lambda c: c.Q + 1.0 * c.P * math.sqrt(n.N) / (1 + c.N))

@torch.no_grad()
def nn_eval(batch: List[Node], net: PolicyValueNet):
    dev = next(net.parameters()).device
    x = torch.stack([n.g.state_tensor() for n in batch]).to(dev)
    logit, v = net(x); logit, v = logit.cpu().numpy(), v.squeeze(1).cpu().numpy()
    for n, lgt, val in zip(batch, logit, v):
        n.val = float(val)
        if not n.unexp: continue
        idx = [m[0] * CFG.board_size + m[1] for m in n.unexp]
        probs = softmax_np(lgt[:CFG.board_size**2][idx])
        for m, p in zip(n.unexp, probs):
            child = Node(n.g.copy(), n, m); child.P = p; n.ch[m] = child
        n.unexp.clear()

def backup(n: Node, v: float):
    while n:
        n.N += 1; n.W += v; n.Q = n.W / n.N; v = -v; n = n.p

def mcts(game: GoGame, net: PolicyValueNet, sims=CFG.mcts_sims, temp=1.0):
    root = Node(game.copy()); nn_eval([root], net)
    for _ in range(sims):
        leaf = select(root); nn_eval([leaf], net); backup(leaf, leaf.val)
    mv, vis = zip(*((m, ch.N) for m, ch in root.ch.items()))
    v = np.array(vis, np.float32)
    probs = np.zeros_like(v)
    if temp < 1e-3: probs[np.argmax(v)] = 1.0
    else: w = v ** (1 / temp); probs = w / w.sum()
    return dict(zip(mv, probs))

# =============================================================#
#                    SELF-PLAY  (CPU)                           #
# =============================================================#
def self_play_worker(args):
    seed, model_path = args    
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    net = PolicyValueNet(CFG.board_size).cpu()
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    g, data = GoGame(CFG.board_size), []
    while not g.game_over():
        pi = mcts(g, net, CFG.mcts_sims, 1.0)
        vec = np.zeros(CFG.board_size**2, np.float32)
        for mv, p in pi.items():
            vec[mv[0] * CFG.board_size + mv[1]] = p
        data.append((g.board.copy(), g.current_player, vec))
        mv = random.choices(list(pi.keys()), weights=list(pi.values()))[0]
        g.make_move(*mv)
    w = g.winner()
    return [(b, p, v, 0 if w == 0 else (1 if p == w else -1)) for b, p, v in data]

# =============================================================#
#                       REPLAY BUFFER                           #
# =============================================================#
class Replay:
    def __init__(self, cap): self.data, self.cap = [], cap
    def add(self, new): self.data.extend(new); self.data = self.data[-self.cap:]
    def sample(self, n): return self.data if n >= len(self.data) else random.sample(self.data, n)

RB = Replay(CFG.replay_cap)

# =============================================================#
#                        TRAINING                               #
# =============================================================#
def to_planes(board, player):
    return torch.from_numpy(np.stack([(board == player).astype(np.float32),
                                      (board == 3 - player).astype(np.float32)], 0))

def batch_tensor(batch):
    bds, pl, pi, z = zip(*batch)
    X = torch.stack([to_planes(b, p) for b, p in zip(bds, pl)]).to(device_gpu)
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
        for s in bar:
            samp = idx[s:s+CFG.train_batch]
            inp, tgt_pi, tgt_z = X[samp], pi_t[samp], z_t[samp]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=CFG.amp):
                out_p, out_v = net(inp)
                out_p = out_p[:, :CFG.board_size**2]
                loss_p = -(tgt_pi * F.log_softmax(out_p, 1)).sum(1).mean()
                loss_v = F.mse_loss(out_v, tgt_z)
                loss = loss_p + loss_v
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        logging.info("Iter %d | Epoch %d | loss %.4f", it_idx, ep+1, loss.item())
    net.eval()

# =============================================================#
#                           MAIN                                #
# =============================================================#
def main():
    print("OceanGo – CUDA:", torch.cuda.is_available(), "AMP:", CFG.amp)
    torch.backends.cudnn.benchmark = True
    net = PolicyValueNet().to(device_gpu)

    ckpt = root / "policy_value_net.pth"
    if ckpt.exists():
        saved = torch.load(ckpt, map_location=device_gpu)
        cur = net.state_dict()
        match = {k: v for k, v in saved.items() if k in cur and v.shape == cur[k].shape}
        cur.update(match); net.load_state_dict(cur, strict=False)
        print(f"Loaded {len(match)}/{len(cur)} tensors.")
    else:
        print("No checkpoint – fresh model.")

    tmp = root / "_tmp_cpu.pth"
    torch.save(net.cpu().state_dict(), tmp)  # workers use CPU copy
    net.to(device_gpu)

    outer = tqdm(range(1, CFG.n_iter + 1), desc="Iterations")
    for it in outer:
        seeds = [random.randrange(2**32) for _ in range(CFG.games_per_iter)]
        with mp.Pool(CFG.num_workers) as pool:
            bar = tqdm(total=CFG.games_per_iter,
                       desc=f"Iter {it} self-play", leave=False)
            for res in pool.imap_unordered(self_play_worker,
                                           [(s, tmp.as_posix()) for s in seeds]):
                RB.add(res); bar.update()
            bar.close()
        train(net, RB.sample(1024), it)
        torch.save(net.state_dict(), ckpt)
        torch.save(net.cpu().state_dict(), tmp)
        net.to(device_gpu)

    # ---- Optional: GUI ----
    # run_gui(net)

# ──────────────────────── entrypoint ───────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
