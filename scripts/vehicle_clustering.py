# 1) Read csv
import pandas as pd
import numpy as np
from pathlib import Path

csvs = [f"veh_outputs_batch_{i}.csv" for i in range(8)]
usecols = ["camera_id","timestamp","total_vehicles","vehicles_per_mpx"] 
dfs = []
base_dir = Path("../result")
for fn in csvs:
    p = base_dir / fn
    df = pd.read_csv(p, usecols=usecols)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# 2) Timestamp Parsing and Cleaning
data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", infer_datetime_format=True)
data["total_vehicles"] = pd.to_numeric(data["total_vehicles"], errors="coerce")
data["vehicles_per_mpx"] = pd.to_numeric(data["vehicles_per_mpx"], errors="coerce")
data = data.dropna(subset=["timestamp","total_vehicles","vehicles_per_mpx"]).copy()

# 3) Time characteristics 
data["hour"] = data["timestamp"].dt.hour
data["weekday"] = data["timestamp"].dt.weekday  # 0=Mon
# Periodic Encoding 
data["hour_sin"] = np.sin(2*np.pi*data["hour"]/24.0)
data["hour_cos"] = np.cos(2*np.pi*data["hour"]/24.0)
data["weekday_sin"] = np.sin(2*np.pi*data["weekday"]/7.0)
data["weekday_cos"] = np.cos(2*np.pi*data["weekday"]/7.0)

# 4) Clustering Feature Matrix
feature_cols = ["total_vehicles","vehicles_per_mpx","hour_sin","hour_cos","weekday_sin","weekday_cos"]
X = data[feature_cols].values

# Big Data Sampling Clustering (Speed-Friendly)
MAX_ROWS = 50000
if len(data) > MAX_ROWS:
    samp = data.sample(n=MAX_ROWS, random_state=42)
    X_samp = samp[feature_cols].values
else:
    samp = data
    X_samp = X

# Standerliazation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_samp_scaled = scaler.fit_transform(X_samp)

# 5) Select k (test both options: k=3 and k=4, then choose one based on the contour coefficient)）
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

def try_k(k):
    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, max_iter=200, random_state=42)
    labels = km.fit_predict(X_samp_scaled)
    sil = silhouette_score(X_samp_scaled, labels)
    return km, labels, sil

cand = []
for k in [3,4]:
    km, labels, sil = try_k(k)
    cand.append((k, km, labels, sil))

best_k, best_km, best_labels, best_sil = sorted(cand, key=lambda x: x[3], reverse=True)[0]
print(f"[INFO] best k = {best_k}, silhouette = {best_sil:.3f}")

# 6) Apply the optimal model to label the entire dataset
X_all_scaled = scaler.transform(X)
all_labels = best_km.predict(X_all_scaled)
data["cluster"] = all_labels

# 7) Interpret the cluster as a semantic label (sorted by “congestion intensity”)
cluster_stats = (
    data.groupby("cluster")
        .agg(n=("cluster","size"),
             mean_total_vehicles=("total_vehicles","mean"),
             mean_vehicles_per_mpx=("vehicles_per_mpx","mean"),
             peak_hour=("hour", lambda s: s.value_counts().idxmax()))
        .sort_values(["mean_vehicles_per_mpx","mean_total_vehicles"], ascending=False)
        .reset_index()
)
# Highest average → Severe, and so on
levels = ["Severe","Moderate","Light","Very Light"]
label_map = {row.cluster: (levels[i] if i < len(levels) else f"Level{i+1}")
             for i, row in cluster_stats.reset_index(drop=True).iterrows()}
data["cluster_label"] = data["cluster"].map(label_map)

# 8) Distribution and Proportion of Camera Dimensions
per_cam = data.groupby(["camera_id","cluster_label"]).size().unstack(fill_value=0)
per_cam["total"] = per_cam.sum(axis=1)
for c in per_cam.columns:
    if c != "total":
        per_cam[c+"_pct"] = (per_cam[c] / per_cam["total"]).round(3)

# 9) Save the output for downstream use
out_dir = Path("result_ml"); out_dir.mkdir(exist_ok=True)
data.sort_values(["camera_id","timestamp"]).to_csv(out_dir/"clustered_traffic.csv", index=False)
cluster_stats.to_csv(out_dir/"cluster_stats.csv", index=False)
per_cam.reset_index().to_csv(out_dir/"per_camera_distribution.csv", index=False)

print(f"[OK] saved to {out_dir}/")
