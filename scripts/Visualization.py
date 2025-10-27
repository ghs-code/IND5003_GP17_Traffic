# -*- coding: utf-8 -*-
"""
时间维度可视化 - 交通模式聚类结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Read the clustering results
file_path = Path(__file__).resolve().parent / "result_ml" / "clustered_traffic.csv"
df = pd.read_csv(file_path, parse_dates=["timestamp"])

# 2. Extract time features
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday
weekday_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
df["weekday_name"] = df["weekday"].map(weekday_map)

order = ["Very Light", "Light", "Moderate", "Severe"]

# 3. Visualization 1：Hourly Aggregation (All-Day Distribution)
plt.figure(figsize=(10,6))
sns.countplot(data=df, x="hour", hue="cluster_label", hue_order=order, palette="RdYlGn_r")
plt.title("Hourly Distribution of Traffic States")
plt.xlabel("Hour of Day (SGT)")
plt.ylabel("Snapshot Count")
plt.legend(title="Traffic State", loc="upper right")
plt.tight_layout()
plt.show()

# 4. Visualization 2: Aggregated by Week (Weekly Mode)
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="weekday_name", hue="cluster_label", hue_order=order, palette="RdYlGn_r")
plt.title("Weekday Distribution of Traffic States")
plt.xlabel("Day of Week")
plt.ylabel("Snapshot Count")
plt.tight_layout()
plt.show()

# 5. Visualization 3: Heatmap (Hour × Week)
# Calculate the Severe proportion per hour per week
pivot = (
    df.groupby(["weekday_name","hour","cluster_label"]).size()
      .reset_index(name="count")
      .pivot_table(index=["weekday_name","hour"], columns="cluster_label", values="count", fill_value=0)
      .reindex(columns=order)
)
# Calculate the serve proportion
pivot["Severe_ratio"] = pivot["Severe"] / pivot.sum(axis=1)

heat_data = pivot["Severe_ratio"].unstack(level=0)
plt.figure(figsize=(10,6))
sns.heatmap(heat_data, cmap="Reds", linewidths=0.3)
plt.title("Heatmap of Severe Congestion Ratio by Hour & Weekday")
plt.xlabel("Weekday")
plt.ylabel("Hour of Day")
plt.tight_layout()
plt.show()

# 6. Visualization 4: Average number of vehicles per cluster over time
plt.figure(figsize=(10,6))
mean_by_hour = df.groupby(["cluster_label","hour"])["total_vehicles"].mean().reset_index()
sns.lineplot(data=mean_by_hour, x="hour", y="total_vehicles", hue="cluster_label", hue_order=order, palette="RdYlGn_r")
plt.title("Average Vehicle Count per Hour by Cluster")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Total Vehicles")
plt.tight_layout()
plt.show()
