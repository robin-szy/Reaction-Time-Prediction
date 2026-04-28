import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# Libraries that are not in standard requirements.txt
from scipy.spatial import ConvexHull


# -------------------------
# Arguments for parsing
# (Mostly used for testing on HPC)
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--metadata-file", default="scanpaths_metadata.csv")
    parser.add_argument("--scanpath-dir", default="scanpaths")
    parser.add_argument("--model-file", default="model.pth")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seq-max-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=523)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--loss", type=str, default="huber",
                        choices=["huber", "mse", "smoothl1"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    return parser.parse_args()


# -----------------
# General functions
# -----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_std(x):
    """
    Just to be robust and avoid any errors.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) <= 1:
        return 0.0
    return float(np.std(x))


def safe_slope(values):
    """
    Just to be robust and avoid any errors.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 1:
        return 0.0
    t = np.arange(len(values), dtype=np.float32)
    return float(np.polyfit(t, values, 1)[0])


def duration_switches(durations, band=0.20):
    # Todo: Test
    durations = np.asarray(durations, dtype=np.float32)

    if len(durations) < 2:
        return 0

    lower_q = 0.5 - band / 2
    upper_q = 0.5 + band / 2

    lower = np.quantile(durations, lower_q)
    upper = np.quantile(durations, upper_q)
    median = np.median(durations)

    states = []
    current_state = int(durations[0] > median)

    for d in durations:
        if d < lower:
            current_state = 0
        elif d > upper:
            current_state = 1

        states.append(current_state)

    states = np.asarray(states, dtype=np.int32)
    switches = np.sum(states[1:] != states[:-1])

    return int(switches)


def compute_grid_cells(x, y, grid_size=10):
    # If 10x10 grid: 100 cells. Then, x and y are in [0, 1], so
    # we can do int(x * 10) to get cell, e.g. 3.4 -> cell 3.
    # Return two positions x and y in single value by 10*y + x, which gives coords
    x_idx = np.clip((x * grid_size).astype(int), 0, grid_size - 1)
    y_idx = np.clip((y * grid_size).astype(int), 0, grid_size - 1)
    return y_idx * grid_size + x_idx


def revisit_features(x, y, grid_size=10):
    # Todo: Test
    cells = compute_grid_cells(x, y, grid_size)
    total_visits = len(cells)
    unique_cells = len(np.unique(cells))
    revisits = total_visits - unique_cells
    revisit_ratio = revisits / (total_visits + 1e-6)
    coverage_ratio = unique_cells / (grid_size * grid_size)

    return revisit_ratio, coverage_ratio


def spatial_entropy(x, y, grid_size=10):
    # Todo: test
    cells = compute_grid_cells(x, y, grid_size)
    counts = np.bincount(cells, minlength=grid_size * grid_size).astype(float)

    p = counts / (counts.sum() + 1e-6)
    p = p[p > 0]

    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(grid_size * grid_size)

    return float(entropy / (max_entropy + 1e-6))    # Normed spatial entropy


def convex_hull_features(x, y):
    """
    In 2D, convex hull gives the smallest convex set containing all points.
    We output the area here to get a graps of the search radius of the subject.
    Read more:
    https://medium.com/@mikatal/the-convex-hull-problem-74875bfbbd6a
    """
    points = np.column_stack([x, y])

    if len(points) < 3:
        return 0.0

    try:
        hull = ConvexHull(points)
        area = hull.volume   # In 2D, scipy stores polygon area in .volume
    except:
        area = 0.0

    return float(area)


def direction_consistency_features(dx, dy):

    step_len = np.sqrt(dx**2 + dy**2)
    valid = step_len > 1e-8

    dx = dx[valid]
    dy = dy[valid]
    step_len = step_len[valid]

    if len(step_len) < 2:
        return 0.0

    ux = dx / step_len
    uy = dy / step_len

    cos_sim = ux[:-1] * ux[1:] + uy[:-1] * uy[1:]
    direction_consistency = np.mean(cos_sim)

    angles = np.arctan2(dy, dx)
    angles_unwrapped = np.unwrap(angles)
    turns = np.diff(angles_unwrapped)

    return float(np.std(angles_unwrapped))




def read_sequence(path, seq_max_len):

    """
    Insights from data exploration
        * Intuitively: Distance traveled with the eyes could be very indicative. When I search for something (e.g. "the" in a text), my eyes jump a lot around the screen until I find it. Then, when I found it, the last step is quite small. Then you stay on it (longer fixated)
    * Distance:
        * Large jumps → exploration
        * Small jumps → focused inspection
        * Early large jumps + later small jumps = efficient
        * Random jumps throughout = inefficient
    * Time:
        * More randomness in eye movement -> More reaction time
        * Short fixations = scanning / uncertainty
        * Many long fixations → difficult task → slower RT
        * Very short noisy fixations → confusion → slower RT
    * Revisiting same region: Uncertainty -> Longer RT
    * Spatial coverage
        * Good search: structured coverage (grid-like, systematic)
        * Bad search: clustered or chaotic
    * Behavior over time:
        * Early chaos → later focus → fast RT
        * Always chaotic → slow RT
    * 46 occurrences where reaction time is 0.0 -> Always length of sequence = 1
    * Distance and duration are heavily right-skewed. Using np.log1p helps to stretch them out a bit more.
    * Histogram of sine(theta) and cosine(theta) show peaks at 0 and +-1, which suggests that eye movements are not completely random, but are more horizontal or vertical scans (structures).
    """

    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.upper().strip() for c in df.columns]

    # During data exploration, I found that duration is heavily right-skewed -> log-transform
    x = df["FPOGX"].to_numpy(dtype=np.float32)
    y = df["FPOGY"].to_numpy(dtype=np.float32)
    dur = df["FPOGD"].to_numpy(dtype=np.float32)
    dur_log = np.log1p(dur)

    # During data exploration, I found that dist is heavily right-skewed -> log-transform
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dist = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)
    dist_log = np.log1p(dist)

    angle = np.arctan2(dy, dx)
    angle_sin = np.sin(angle).astype(np.float32)
    angle_cos = np.cos(angle).astype(np.float32)
    angle_unwrapped = np.unwrap(angle)
    dangle = np.diff(angle_unwrapped, prepend=angle_unwrapped[0]).astype(np.float32)

    seq = np.stack(
        [
            x,
            y,
            dx,
            dy,
            dist_log,
            angle_sin,
            angle_cos,
            dangle,
            dur_log,
        ],
        axis=1,
    ).astype(np.float32)


    # Global features

    # Length of sequence
    n = len(df) # Todo: Maybe remove, because dur_sum might be even better. Corr 0.89% between the two, so I keep it for now.
    n_log = np.log1p(n)     # Also skewed

    # Distance features
    # First element of dist always 0
    dist_no_first = dist[1:] if n > 1 else np.array([], dtype=np.float32)

    if len(dist_no_first) > 0:
        dist_sum = float(dist_no_first.sum())
        dist_max = float(dist_no_first.max())
        #large_threshold = np.percentile(dist_no_first, 75)
        #num_large_jumps = float((dist_no_first > large_threshold).sum())   # Was a top feature, but 99% correlation with n, so redundant
        dist_trend = safe_slope(dist_no_first)
        first_third_end = max(1, len(dist_no_first) // 3)
        dist_first_third = float(dist_no_first[:first_third_end].mean())
    else:
        dist_sum = dist_max = dist_trend = dist_first_third = 0.0

    # Duration features
    dur_diff = np.abs(np.diff(dur)) if n > 1 else np.array([], dtype=np.float32)
    dur_switches = duration_switches(dur, band=0.20)

    # Spatial features
    revisit_ratio, coverage_ratio = revisit_features(x, y, grid_size=10)
    spatial_angle_std = direction_consistency_features(dx, dy)
    spatial_normed_entropy = spatial_entropy(x, y, grid_size=10)
    hull_area = convex_hull_features(x, y)

    global_features = np.array(
        [
            float(n_log),
            # Distance features
            #dist_sum,
            np.log1p(dist_sum), # Also skewed
            dist_trend,
            dist_first_third,
            dist_max,
            # Duration features
            #float(dur.sum()),
            np.log1p(float(dur.sum())), # Right-skewed
            np.log1p(float(dur.max())), # Slightly right-skewed, benefits from that
            safe_std(dur_diff),
            dur_switches,
            # Spatial features
            revisit_ratio,
            coverage_ratio,
            spatial_normed_entropy,
            np.log1p(spatial_angle_std),    # Only really skewed one from spatial features
            hull_area
        ],
        dtype=np.float32,
    )

    if len(seq) > seq_max_len:
        idx = np.linspace(0, len(seq) - 1, seq_max_len).astype(int)
        seq = seq[idx]

    return seq, global_features


def load_items(data_dir, metadata_file, scanpath_dir):
    # Todo: Test
    meta_path = os.path.join(data_dir, metadata_file)
    scan_dir = os.path.join(data_dir, scanpath_dir)
    meta = pd.read_csv(meta_path, sep=r"\s+", header=None, names=["reaction_time", "filename"], engine="python")
    items = []
    for _, row in meta.iterrows():
        path = os.path.join(scan_dir, str(row["filename"]))
        if os.path.exists(path):
            items.append((path, float(row["reaction_time"])))
    if not items:
        raise ValueError(f"No usable scanpaths found from {meta_path} and {scan_dir}")
    return items


def compute_global_norm(items, seq_max_len):
    """
    Helper function to normalize the global features
    """
    features = []
    for path, _ in items:
        _, g = read_sequence(path, seq_max_len)
        features.append(g)
    features = np.stack(features)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6

    return mean.astype(np.float32), std.astype(np.float32)


def compute_seq_norm(items, seq_max_len):
    """
    Compute mean/std for sequence features using only the training set.
    """
    features = []
    for path, _ in items:
        seq, _ = read_sequence(path, seq_max_len)
        features.append(seq)
    features = np.concatenate(features, axis=0)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6

    return mean.astype(np.float32), std.astype(np.float32)


class ScanpathDataset(Dataset):
    # The same as in first homework
    def __init__(self, items, seq_max_len, global_mean=None, global_std=None, seq_mean=None, seq_std=None):
        self.items = items
        self.seq_max_len = seq_max_len
        self.global_mean = global_mean
        self.global_std = global_std
        self.seq_mean = seq_mean
        self.seq_std = seq_std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        seq, global_features = read_sequence(path, self.seq_max_len)

        if self.seq_mean is not None:
            seq = (seq - self.seq_mean) / self.seq_std

        if self.global_mean is not None:    # Normalize global features
            global_features = (global_features - self.global_mean) / self.global_std

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(global_features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def collate_batch(batch):

    seqs, global_attr, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()

    feat_dim = seqs[0].shape[1]
    x = torch.zeros(len(seqs), max_len, feat_dim)
    for i, seq in enumerate(seqs):
        x[i, : seq.shape[0]] = seq

    g = torch.stack(global_attr)
    y = torch.stack(labels)
    return x, lengths, g, y


# -----------------
# Model
# -----------------
class HybridGRURegressor(nn.Module):
    # The input sizes so I don't have to adapt the size every time when I add/remove a feature
    def __init__(self, hidden_size, seq_input_size, global_input_size, dropout=0.05):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size + global_input_size),
            nn.Linear(hidden_size + global_input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.out = nn.Softplus()

    def forward(self, x, lengths, global_features):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h_last = h[-1]
        combined = torch.cat([h_last, global_features], dim=1)
        raw = self.fc(combined).squeeze(1)
        return self.out(raw)    # Softplus


# -----------------
# Training
# -----------------
def train(args):

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_items(args.data_dir, args.metadata_file, args.scanpath_dir)
    random.shuffle(items)
    split = int((1.0 - args.val_frac) * len(items))
    train_items = items[:split]
    val_items = items[split:]

    if not train_items or not val_items:
        raise ValueError("Train/validation split failed. Check data size and --val-frac.")

    # Global means for normalization
    global_mean, global_std = compute_global_norm(train_items, args.seq_max_len)
    seq_mean, seq_std = compute_seq_norm(train_items, args.seq_max_len)

    train_dataset = ScanpathDataset(
        train_items,
        seq_max_len=args.seq_max_len,
        global_mean=global_mean,
        global_std=global_std,
        seq_mean=seq_mean,
        seq_std=seq_std
    )

    val_dataset = ScanpathDataset(
        val_items,
        seq_max_len=args.seq_max_len,
        global_mean=global_mean,
        global_std=global_std,
        seq_mean=seq_mean,
        seq_std=seq_std
    )

    # Load datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    # Todo: Test if really len(seq_mean)
    model = HybridGRURegressor(
        hidden_size=args.hidden_size,
        seq_input_size=len(seq_mean),
        global_input_size=len(global_mean),
        dropout=args.dropout,
    ).to(device)

    # How many model parameters do I have?
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print("Parameters per layer:")
    for name, p in model.named_parameters():
        print(name, p.numel())

    # Loss function
    # Since the label is bimodal, penalizing large errors by least squares may be bad.
    # Like a bad prediction to second mode at 10s could badly influence gradients
    # But on other hand, MAE or similar is robust, but the gradients are less smoth
    # Huber combines both worlds
    # Source: https://medium.com/@mlblogging.k/14-loss-functions-you-can-use-for-regression-b24db8dff987
    # On the other hand, evaluation is via RMSE.
    #loss_fn = nn.SmoothL1Loss()
    if args.loss == "mse":
        loss_fn = nn.MSELoss()
    elif args.loss == "huber":
        loss_fn = nn.HuberLoss(delta=args.huber_delta)

    # Optimizer
    if args.weight_decay == 0.0:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    best_rmse = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # Todo: Test

    for epoch in range(args.epochs):
        model.train()
        total_train_loss, total_train = 0.0, 0

        for x, lengths, g, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            g = g.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x, lengths, g)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item()) * len(y)
            total_train += len(y)

        train_loss = total_train_loss / max(total_train, 1)

        val_loss, val_rmse = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn
        )

        improved = val_rmse < best_rmse - args.min_delta
        if improved:
            best_rmse = val_rmse
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_rmse={val_rmse:.4f} "
            f"best_rmse={best_rmse:.4f}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}. "
                  f"Best val RMSE={best_rmse:.4f}"
                  )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "seq_max_len": args.seq_max_len,
        "seq_input_size": int(len(seq_mean)),
        "global_input_size": int(len(global_mean)),
        "global_mean": torch.tensor(global_mean, dtype=torch.float32),
        "global_std": torch.tensor(global_std, dtype=torch.float32),
        "seq_mean": torch.tensor(seq_mean, dtype=torch.float32),
        "seq_std": torch.tensor(seq_std, dtype=torch.float32),
        "huber_delta": args.huber_delta,
    }

    torch.save(checkpoint, args.model_file)
    print(f"Saved best model to {args.model_file} with RMSE={best_rmse:.4f}")

    # Summary for HPC testing (to not open every log file every time)
    results_file = "runs/results.csv"
    row = {
        "model_file": args.model_file,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seq_max_len": args.seq_max_len,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "seed": args.seed,
        "best_rmse": best_rmse,
        "total_params": total_params,
        "huber_delta": args.huber_delta,
    }

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(results_file):
        df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df.to_csv(results_file, index=False)


# -----------------
# Evaluation
# -----------------
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, total_sq_error, total = 0.0, 0.0, 0

    with torch.no_grad():
        for x, lengths, g, y in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            g = g.to(device)
            y = y.to(device)

            # Prediction
            pred = model(x, lengths, g)

            # Loss
            loss = loss_fn(pred, y)
            total_loss += float(loss.item()) * len(y)
            # Todo: Test if corresponds to formula of RMSE
            total_sq_error += float(torch.sum((pred - y) ** 2).item())
            total += len(y)

    avg_loss = total_loss / max(total, 1)
    rmse = np.sqrt(total_sq_error / max(total, 1))
    return avg_loss, rmse



def load_and_predict(directory, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)

    hidden_size = int(checkpoint["hidden_size"])
    dropout = float(checkpoint.get("dropout", 0.0))
    seq_input_size = int(checkpoint["seq_input_size"])
    global_input_size = int(checkpoint["global_input_size"])

    model = HybridGRURegressor(
        hidden_size=hidden_size,
        seq_input_size=seq_input_size,
        global_input_size=global_input_size,
        dropout=dropout
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    seq_max_len = int(checkpoint["seq_max_len"])
    global_mean = checkpoint["global_mean"].cpu().numpy()
    global_std = checkpoint["global_std"].cpu().numpy()
    seq_mean = checkpoint["seq_mean"].cpu().numpy()
    seq_std = checkpoint["seq_std"].cpu().numpy()

    pred_dict = {}
    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))

    with torch.no_grad():
        for path in paths:
            # The following function read_sequence performs the two first steps required in eval.py:
            # (1) Read the data from the provided directory
            # (2) Prepare the data according to preprocessing pipeline of model training
            seq, global_features = read_sequence(path, seq_max_len)
            seq = (seq - seq_mean) / seq_std
            global_features = (global_features - global_mean) / global_std

            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([len(seq)], dtype=torch.long).to(device)
            g = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(x, lengths, g).item()
            pred_dict[os.path.abspath(path)] = float(pred)

    return pred_dict


if __name__ == "__main__":
    train(parse_args())
