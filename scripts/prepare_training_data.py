import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from datetime import timedelta
from sklearn.model_selection import train_test_split  

'''Prepare training data and generate datasets for two road segments.
This script loads data from clustered_traffic.csv
splits them into two datasets representing different road segments. 

Workflow:
    1. Load CSV & Clean
    2. Multi-camera -> Single road segment sequence
    3. Sliding window (with multiple features)
    4. Randomly split train/val 
    5. Save as npz&json file

'''

CSV_PATH = "scripts/result_ml/clustered_traffic.csv"   
CAUSEWAY_CAMS    = [2701, 2702, 2704, 2706]
SECOND_LINK_CAMS = [4703, 4707, 4712, 4713]

FREQ_MINUTES = 5

T_IN  = 24   # Sequence length
T_OUT = 6    # Predicted step size

TRAIN_RATIO = 0.8  

OUT_DIR = "scripts/artifacts/phase_cluster"
os.makedirs(OUT_DIR, exist_ok=True)


def load_cluster_csv(csv_path):
    """
    Read camera-level clustering results:
    Required columns: camera_id, timestamp, cluster_label
    Generated: label_num (1..4)
    """

    df = pd.read_csv(csv_path)

    required_cols = ["camera_id", "timestamp", "cluster_label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} Missing culumns: {missing}")

    df = df[["camera_id", "timestamp", "cluster_label"]].copy()

    # Unified time
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Text -> Number Labels
    label_map = {
        "very light": 1, "very_light": 1, "Very Light": 1,
        "light": 2, "Light": 2,
        "moderate": 3, "Moderate": 3,
        "severe": 4, "Severe": 4,
    }
    df["label_num"] = df["cluster_label"].map(label_map)

    df = df.dropna(subset=["timestamp", "label_num"]).copy()
    df["label_num"] = df["label_num"].astype(int)

    return df



def majority_vote(values):
    """
    Majority vote: Returns the label that appears most frequently.
    In case of a tie, the more severe (larger) label is used.
    """
    cnt = Counter(values)
    max_count = max(cnt.values())
    cands = [val for val, c in cnt.items() if c == max_count]
    return max(cands)

def build_segment_series(df_all, cam_list, freq_minutes):
    """
    Input:
        df_all: All cameras (level per cam per timestamp)
        cam_list: Cameras included in this road segment
    Output:
        seg_df: index = timestamp (equal intervals of 5 minutes)
        columns = [route_level, hour_sin, hour_cos, weekday_sin, weekday_cos] 
    Steps:
        1. Filter cameras
        2. For the same timestamp, perform majority voting on all camera label_num -> route_level
        3. Generate a complete time index and impute it forward
        4. Calculate time features based on timestamps and concatenate them into a DataFrame 
    """

    sub = df_all[df_all["camera_id"].isin(cam_list)].copy()
    if sub.empty:
        raise ValueError(f"No data from camrea: {cam_list}")

    # Timestamp aggregation -> Congestion level of this road segment at this moment
    grouped = (
        sub.groupby("timestamp")["label_num"]
           .apply(lambda x: majority_vote(list(x)))
           .sort_index()
           .to_frame(name="route_level")
    )

    # Complete into an equally spaced timeline
    t_min = grouped.index.min()
    t_max = grouped.index.max()
    full_index = pd.date_range(start=t_min, end=t_max, freq=f"{freq_minutes}min")

    seg_df = grouped.reindex(full_index)

    # Forward filling, then add 1 to the beginning (lightest method)
    seg_df["route_level"] = seg_df["route_level"].ffill()
    seg_df["route_level"] = seg_df["route_level"].fillna(1).astype(int)

    #  Time characteristics: based on index full_index
    ts_idx = seg_df.index

    hours   = ts_idx.hour.astype(float)
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)

    weekdays = ts_idx.weekday.astype(float)  # Monday=0 ... Sunday=6
    weekday_sin = np.sin(2 * np.pi * weekdays / 7.0)
    weekday_cos = np.cos(2 * np.pi * weekdays / 7.0)

    seg_df["hour_sin"] = hour_sin
    seg_df["hour_cos"] = hour_cos
    seg_df["weekday_sin"] = weekday_sin
    seg_df["weekday_cos"] = weekday_cos

    seg_df.index.name = "timestamp"
    return seg_df
    # columns: route_level(int), hour_sin(float), hour_cos(float), weekday_sin(float), weekday_cos(float)




def make_sliding_windows_multifeat(df_series, t_in, t_out):
    """
    df_series: DataFrame, index=timestamp, columns:
        route_level, hour_sin, hour_cos, weekday_sin, weekday_cos
    Construct supervised learning samples:
        X: Past t_in steps, each step contains these 5 features
            [route_level(1..4), hour_sin, hour_cos, weekday_sin, weekday_cos]
        y: Future t_out steps' route_level (classification labels 1..4)
    Returns:
        X (num_samples, t_in, num_features=5)
        y (num_samples, t_out)
        t_starts (num_samples,) 
    """

    feature_cols = ["route_level", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
    arr_feat = df_series[feature_cols].values  # shape [T,5]
    arr_label = df_series["route_level"].values  # shape [T]

    X_list, Y_list, T_list = [], [], []
    total = len(df_series)
    need = t_in + t_out

    for start in range(total - need + 1):
        hist_feat = arr_feat[start : start + t_in]          # (t_in,5)
        fut_label = arr_label[start + t_in : start + t_in + t_out]  # (t_out,)

        X_list.append(hist_feat)
        Y_list.append(fut_label)
        T_list.append(df_series.index[start])

    if not X_list:
        return None, None, None

    X = np.stack(X_list, axis=0)  # (N, t_in, 5)
    Y = np.stack(Y_list, axis=0)  # (N, t_out)
    T0 = np.array(T_list)
    return X, Y, T0



def random_train_val_split(X, Y, test_ratio):
    """
    Use sklearn's `train_test_split` function to randomly split the data.
    Note: We only split X and Y, not in chronological order.    
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=test_ratio, shuffle=True, random_state=42
    )
    return X_train, y_train, X_val, y_val

def save_outputs(prefix, Xtr, ytr, Xva, yva, meta):
    npz_path = os.path.join(OUT_DIR, f"{prefix}_dataset.npz")
    json_path = os.path.join(OUT_DIR, f"{prefix}_meta.json")

    np.savez_compressed(
        npz_path,
        X_train=Xtr, y_train=ytr,
        X_val=Xva,   y_val=yva,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[saved] {npz_path}")
    print(f"[saved] {json_path}")

    return {
        "dataset_npz": npz_path,
        "meta_json": json_path
    }




def build_segment_dataset(seg_name, cam_list, all_df):
    """
    seg_name: "causeway" / "second_link"
    cam_list: List of camera IDs
    all_df: Full camera-level data
    """

    seg_df = build_segment_series(all_df, cam_list, FREQ_MINUTES)
    # seg_df columns:
    #   route_level(int), hour_sin, hour_cos, weekday_sin, weekday_cos

    X_all, Y_all, T0 = make_sliding_windows_multifeat(seg_df, T_IN, T_OUT)
    if X_all is None:
        print(f"[{seg_name}] Acruire more data")
        return None

    X_train, y_train, X_val, y_val = random_train_val_split(
        X_all, Y_all, test_ratio=(1.0 - TRAIN_RATIO)
    )
    meta = {
        "segment": seg_name,
        "camera_ids": cam_list,
        "csv_path": CSV_PATH,
        "freq_minutes": FREQ_MINUTES,
        "t_in": T_IN,
        "t_out": T_OUT,
        "train_ratio_random": TRAIN_RATIO,  
        "num_total_points_after_resample": int(len(seg_df)),
        "num_samples_all": int(len(X_all)),
        "num_samples_train": int(len(X_train)),
        "num_samples_val": int(len(X_val)),
        "X_train_shape": list(np.shape(X_train)),  # (N_tr, 24, 5)
        "y_train_shape": list(np.shape(y_train)),  # (N_tr, 6)
        "X_val_shape":   list(np.shape(X_val)),
        "y_val_shape":   list(np.shape(y_val)),
        "feature_order": [
            "route_level(1..4)",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos"
        ],
        "label_definition": {
            "Very Light": 1,
            "Light": 2,
            "Moderate": 3,
            "Severe": 4
        },
        "aggregation": "majority vote over cameras per timestamp; tie -> higher congestion",
        "note_split": "Samples shuffled, then train_test_split(random_state=42)"
    }

    files = save_outputs(
        prefix=f"phase_cluster_{seg_name}",
        Xtr=X_train, ytr=y_train,
        Xva=X_val,   yva=y_val,
        meta=meta
    )

    print(f"[{seg_name}] DONE")
    print(f"  cameras used: {sorted(cam_list)}")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val   shape: {X_val.shape},   y_val   shape: {y_val.shape}")
    return files



if __name__ == "__main__":
    all_df = load_cluster_csv(CSV_PATH)

    causeway_files    = build_segment_dataset("causeway",    CAUSEWAY_CAMS,    all_df)
    second_link_files = build_segment_dataset("second_link", SECOND_LINK_CAMS, all_df)

    print("All done.")
