# Assignment: Reaction time prediction.
# Author Robin Szymanski


import os
import glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

# Libraries that are not in standard requirements.txt
from scipy.spatial import ConvexHull

# -----------------
# General functions
# -----------------

def safe_std(x):
    # Just to be robust and avoid any errors.
    x = np.asarray(x, dtype=np.float32)
    if len(x) <= 1:
        return 0.0
    return float(np.std(x))


def safe_slope(values):
    # Just to be robust and avoid any errors.
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 1:
        return 0.0
    t = np.arange(len(values), dtype=np.float32)
    return float(np.polyfit(t, values, 1)[0])


def duration_switches(durations, band=0.20):

    durations = np.asarray(durations, dtype=float)

    if len(durations) < 2:
        return 0

    lower_q = 0.5 - band / 2
    upper_q = 0.5 + band / 2

    lower = np.quantile(durations, lower_q)
    upper = np.quantile(durations, upper_q)

    states = []

    # Initialize only once a value clearly leaves the hysteresis band
    current_state = None

    for d in durations:
        if d < lower:
            current_state = 0
        elif d > upper:
            current_state = 1
        else:
            # inside hysteresis band
            if current_state is None:
                current_state = 0  # neutral default before first clear signal

        states.append(current_state)

    states = np.asarray(states, dtype=int)
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
    cells = compute_grid_cells(x, y, grid_size)
    total_visits = len(cells)
    unique_cells = len(np.unique(cells))
    revisits = total_visits - unique_cells
    revisit_ratio = revisits / (total_visits + 1e-6)
    coverage_ratio = unique_cells / (grid_size * grid_size)

    return revisit_ratio, coverage_ratio


def spatial_entropy(x, y, grid_size=10):

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
    #direction_consistency = np.mean(cos_sim)

    angles = np.arctan2(dy, dx)
    angles_unwrapped = np.unwrap(angles)
    #turns = np.diff(angles_unwrapped)

    return float(np.std(angles_unwrapped))


def read_sequence(path, seq_max_len):

    """
    Main function creating all the features (with helper functions)

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

    # Sequential features

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

    dt = np.diff(dur, prepend=dur[0])
    speed = dist / (np.abs(dt) + 1e-6)
    speed = np.clip(speed, 0.0, 20.0)
    speed_log = np.log1p(speed)

    angle = np.arctan2(dy, dx)
    angle_sin = np.sin(angle).astype(np.float32)
    angle_cos = np.cos(angle).astype(np.float32)
    angle_unwrapped = np.unwrap(angle)
    dangle = np.diff(angle_unwrapped, prepend=angle_unwrapped[0]).astype(np.float32)
    cum_time = np.cumsum(dur).astype(np.float32)

    seq = np.stack(
        [
            x,
            y,
            #dx,
            #dy,
            dt,
            speed_log,
            dist_log,
            angle_sin,
            angle_cos,
            dangle,
            dur_log,
            cum_time
        ],
        axis=1,
    ).astype(np.float32)


    # Global features

    # Length of sequence
    n = len(df)
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


# -----------------
# Model
# -----------------
class HybridGRURegressor(nn.Module):
    # The input sizes so I don't have to adapt the size every time when I add/remove a feature
    def __init__(self, hidden_size, seq_input_size, global_input_size,
                 dropout=0.05, bidirectional=False, bottleneck_dim=4, global_dropout=0.3):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        gru_output_size = hidden_size * self.num_directions

        self.gru = nn.GRU(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.global_proj = nn.Sequential(
            nn.Linear(global_input_size, bottleneck_dim),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(gru_output_size + bottleneck_dim),
            nn.Linear(gru_output_size + bottleneck_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.global_dropout = nn.Dropout(global_dropout)

        """
        self.fc = nn.Sequential(    # Only sequential
            nn.LayerNorm(gru_output_size),
            nn.Linear(gru_output_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )"""

        """
        self.fc = nn.Sequential(    # Standard global
            nn.LayerNorm(gru_output_size + global_input_size),
            nn.Linear(gru_output_size + global_input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )"""

        self.out = nn.Softplus()

    def forward(self, x, lengths, global_features, global_alpha=1.0):
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h = self.gru(packed)

        if self.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=1)
        else:
            h_last = h[-1]

        global_features = self.global_dropout(global_features)
        g = self.global_proj(global_features)
        g = global_alpha * g
        combined = torch.cat([h_last, g], dim=1)
        raw = self.fc(combined).squeeze(1)

        #raw = self.fc(h_last).squeeze(1)    # Sequential only

        return self.out(raw)


def load_and_predict(directory, model_file):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    checkpoint = torch.load(model_file, map_location=device, weights_only=True)

    hidden_size = int(checkpoint["hidden_size"])
    dropout = float(checkpoint.get("dropout", 0.0))
    seq_input_size = int(checkpoint["seq_input_size"])
    global_input_size = int(checkpoint["global_input_size"])
    bidirectional = bool(checkpoint.get("bidirectional", False))
    bottleneck_dim = int(checkpoint.get("bottleneck_dim", 2))
    global_dropout = float(checkpoint.get("global_dropout", 0.5))
    global_alpha = float(checkpoint.get("global_alpha", 1.0))
    seq_max_len = int(checkpoint["seq_max_len"])
    global_mean = checkpoint["global_mean"].cpu().numpy()
    global_std = checkpoint["global_std"].cpu().numpy()
    seq_mean = checkpoint["seq_mean"].cpu().numpy()
    seq_std = checkpoint["seq_std"].cpu().numpy()

    model = HybridGRURegressor(
        hidden_size=hidden_size,
        seq_input_size=seq_input_size,
        global_input_size=global_input_size,
        dropout=dropout,
        bidirectional=bidirectional,
        bottleneck_dim=bottleneck_dim,
        global_dropout=global_dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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

            # Query the model with data in (1)
            pred = model(x, lengths, g, global_alpha=global_alpha).item()
            pred_dict[os.path.abspath(path)] = float(pred)

    # Return a dictionary where keys are absolute file paths and values are the predicted reaction time for each file
    return pred_dict


# Smoke test
if __name__ == "__main__":
    predictions = load_and_predict("scanpaths/test", "model.pth")
    for path, pred in predictions.items():
        print(f"{path}\t{pred}")