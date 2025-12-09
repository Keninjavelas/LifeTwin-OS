from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from ml.utils.load_export import load_exported_summaries


def build_sequence_dataset(
    min_len: int = 2,
    sliding: bool = False,
    max_windows_per_summary: Optional[int] = None,
) -> Tuple[List[List[int]], List[int], Dict[str, int]]:
    """Build sequences and targets from exported summaries.

    Args:
        min_len: minimum number of apps in a summary to consider.
        sliding: if True, create multiple windows by sliding over top_apps list.
        max_windows_per_summary: limit windows per summary when sliding.

    Returns: (sequences, targets, app_to_id)
    """
    payload = load_exported_summaries()
    summaries = payload.get("summaries", [])
    app_to_id: Dict[str, int] = {}
    sequences: List[List[int]] = []
    targets: List[int] = []

    for s in summaries:
        apps = s.get("top_apps") or []
        ids = []
        for a in apps:
            if a not in app_to_id:
                app_to_id[a] = len(app_to_id) + 1
            ids.append(app_to_id[a])

        if len(ids) < min_len:
            continue

        if sliding and len(ids) > min_len:
            windows = []
            for start in range(0, max(1, len(ids) - 1)):
                inp = ids[: start + (min_len - 1)]
                tgt = ids[start + (min_len - 1)]
                windows.append((inp, tgt))
                if max_windows_per_summary and len(windows) >= max_windows_per_summary:
                    break
            for inp, tgt in windows:
                sequences.append(inp)
                targets.append(tgt)
        else:
            sequences.append(ids[:-1])
            targets.append(ids[-1])

    return sequences, targets, app_to_id


def pad_sequences(X: List[List[int]], pad_value: int = 0, max_len: Optional[int] = None) -> np.ndarray:
    import numpy as _np
    if not X:
        return _np.array([])
    m = max((len(x) for x in X)) if max_len is None else max_len
    padded = [([pad_value] * (m - len(x))) + x for x in X]
    return _np.array(padded, dtype=_np.int64)


def build_time_series_dataset(window: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    payload = load_exported_summaries()
    summaries = payload.get("summaries", [])
    if not summaries:
        return np.array([]), np.array([])
    df = pd.DataFrame(summaries)
    # ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    # use total_screen_time as the target series
    series = df.get("total_screen_time", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
    X = []
    y = []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)
