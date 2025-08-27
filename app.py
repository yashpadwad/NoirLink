"""
Streamlit Dashboard for NoirLink (PPEC)
---------------------------------------
- Loads clustered CDR data (data/cdr_clusters.csv)
- Shows table of users + cluster summary
- Plots scatterplot of clusters
- Verifies blockchain integrity automatically
- Allows new user input for real-time classification
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import hashlib
import json
import os
from web3 import Web3
import numpy as np
from sklearn.metrics import euclidean_distances

# ---------------------------
# Config
# ---------------------------
CSV_PATH = "data/cdr_clusters.csv"
GANACHE_RPC = "http://127.0.0.1:7545"
ABI_FILE = "contract_abi.json"
ADDR_FILE = "contract_address.txt"

# ---------------------------
# Helpers
# ---------------------------
def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            h.update(chunk)
    return "0x" + h.hexdigest()

def load_contract(w3):
    """Load deployed contract if ABI + address exist"""
    if not (os.path.exists(ABI_FILE) and os.path.exists(ADDR_FILE)):
        return None
    with open(ABI_FILE, "r") as f:
        abi = json.load(f)
    with open(ADDR_FILE, "r") as f:
        address = f.read().strip()
    return w3.eth.contract(address=address, abi=abi)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="NoirLink Dashboard", layout="wide")
st.title("🔒 NoirLink: Privacy-Preserving Encrypted Clustering")

# Load CSV
try:
    df = pd.read_csv(CSV_PATH)
    st.success(f"Loaded {len(df)} records from {CSV_PATH}")
except FileNotFoundError:
    st.error("No clustered data found. Run `idpc_encrypted.py` first.")
    st.stop()

# Show data preview
st.subheader("📊 Clustered Call Data")
st.dataframe(df.head(10)) # Showing 10 rows for a cleaner look

# --- NEW: Re-organized Cluster Analytics Section ---
st.subheader("📝 Cluster Analytics")
feature_cols = ['calls_per_day', 'avg_call_duration_sec', 'avg_data_mb_per_day']

try:
    # Use columns for a neat side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.info("This table shows the average profile of each cluster.")
        # Group by cluster and calculate the mean for the feature columns
        cluster_summary = df.groupby('cluster')[feature_cols].mean().round(1)
        cluster_summary.columns = ["Avg Calls/Day", "Avg Duration (sec)", "Avg Data/Day (MB)"]
        st.dataframe(cluster_summary, use_container_width=True)

    with col2:
        st.info("This chart shows the number of users in each cluster.")
        # --- ADDED: Feature 1 - Cluster Size Distribution Plot ---
        cluster_counts = df['cluster'].value_counts().sort_index()
        fig_bar = px.bar(
            cluster_counts, 
            x=cluster_counts.index, 
            y=cluster_counts.values,
            labels={'x': 'Cluster ID', 'y': 'Number of Users'},
            title="Cluster Population"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- ADDED: Feature 2 - Automatic Cluster Descriptions ---
    st.subheader("🗣️ Automatic Cluster Profiles")
    st.info("This section provides a simple, human-readable summary of each cluster's characteristics compared to the overall average.")
    
    overall_avg = df[feature_cols].mean()

    for i, row in cluster_summary.iterrows():
        # Compare cluster average to overall average to generate profile
        calls_profile = "High" if row["Avg Calls/Day"] > overall_avg["calls_per_day"] * 1.1 else "Low"
        duration_profile = "Long" if row["Avg Duration (sec)"] > overall_avg["avg_call_duration_sec"] * 1.1 else "Short"
        data_profile = "Heavy" if row["Avg Data/Day (MB)"] > overall_avg["avg_data_mb_per_day"] * 1.1 else "Light"
        
        st.markdown(f"- **Cluster {i}:** Characterized by **{calls_profile} Calls**, **{duration_profile} Duration**, and **{data_profile} Data Usage**.")

except Exception as e:
    st.warning(f"Could not generate cluster analytics: {e}")
# --- End of new section ---


# Scatterplot
st.subheader("🌀 Cluster Visualization")
x_axis = st.selectbox("X-axis", options=feature_cols, index=0)
y_axis = st.selectbox("Y-axis", options=feature_cols, index=2)

fig = px.scatter(
    df, x=x_axis, y=y_axis, color="cluster",
    hover_data=["user_id"],
    title="User Clusters"
)
st.plotly_chart(fig, use_container_width=True)

# Blockchain verification
st.subheader("🔗 Blockchain Verification")
try:
    file_hash = sha256_of_file(CSV_PATH)
    st.write("Local CSV SHA256:", file_hash)

    w3 = Web3(Web3.HTTPProvider(GANACHE_RPC))
    if w3.is_connected():
        contract = load_contract(w3)
        if contract:
            onchain_hash = contract.functions.storedHash().call()
            onchain_hex = w3.to_hex(onchain_hash)
            st.write("On-chain hash:", onchain_hex)

            if onchain_hex.lower() == file_hash.lower():
                st.success("✅ Blockchain verification SUCCESS: Hashes match")
            else:
                st.error("❌ Blockchain verification FAILURE: Hash mismatch")
        else:
            st.info("ℹ️ No deployed contract found. Run blockchain_verification.py first.")
    else:
        st.warning("⚠️ Could not connect to Ganache. Start Ganache before running.")
except Exception as e:
    st.error(f"Blockchain error: {e}")

# New User Input -> Cluster Assignment + Visualization
st.subheader("📍 Classify a New User")

with st.form("new_user_form"):
    calls = st.number_input("Calls per day", min_value=0, max_value=50, value=10)
    duration = st.number_input("Average call duration (sec)", min_value=0, max_value=1000, value=200)
    data_usage = st.number_input("Average data usage per day (MB)", min_value=0, max_value=10000, value=3000)
    submitted = st.form_submit_button("Classify Me")

if submitted:
    features = df[feature_cols]
    clusters = df["cluster"]

    user_point = np.array([[calls, duration, data_usage]])
    dists = euclidean_distances(user_point, features)
    nearest_idx = dists.argmin()
    user_cluster = clusters.iloc[nearest_idx]

    st.success(f"✅ Based on your input, this user belongs to **Cluster {user_cluster}**")
    st.info(f"Their profile is most similar to existing user: {df.iloc[nearest_idx]['user_id']}")

    fig_new = px.scatter(
        df, x=x_axis, y=y_axis, color="cluster",
        hover_data=["user_id"],
        title=f"Clusters with New User (Belongs to Cluster {user_cluster})"
    )
    
    new_user_data = {
        'calls_per_day': calls,
        'avg_call_duration_sec': duration,
        'avg_data_mb_per_day': data_usage
    }

    fig_new.add_scatter(
        x=[new_user_data[x_axis]], 
        y=[new_user_data[y_axis]],
        mode='markers',
        marker=dict(
            color='red',
            symbol='star',
            size=15,
            line=dict(width=2, color='DarkSlateGrey')
        ),
        name="New User"
    )

    st.plotly_chart(fig_new, use_container_width=True)