"""
Streamlit Dashboard for NoirLink (PPEC)
---------------------------------------
- Loads clustered CDR data (data/cdr_clusters.csv)
- Shows table of users + cluster
- Plots scatterplot of clusters
- Verifies blockchain integrity automatically
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
st.dataframe(df.head(20))

# Scatterplot
st.subheader("🌀 Cluster Visualization")
x_axis = st.selectbox("X-axis", options=df.columns[1:-1], index=0)
y_axis = st.selectbox("Y-axis", options=df.columns[1:-1], index=1)

fig = px.scatter(
    df, x=x_axis, y=y_axis, color="cluster",
    hover_data=["user_id"],
    title="User Clusters"
)
st.plotly_chart(fig, use_container_width=True)

# Blockchain verification
st.subheader("🔗 Blockchain Verification")
file_hash = sha256_of_file(CSV_PATH)
st.write("Local CSV SHA256:", file_hash)

try:
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

# ---------------------------
# New User Input -> Cluster Assignment + Visualization
# ---------------------------
st.subheader("📝 Try Your Own Data")

calls = st.number_input("Calls per day", min_value=0, max_value=50, value=10)
duration = st.number_input("Average call duration (sec)", min_value=0, max_value=1000, value=200)
data_usage = st.number_input("Average data usage per day (MB)", min_value=0, max_value=10000, value=3000)

if st.button("Classify Me"):
    # Extract existing features + clusters
    features = df[["calls_per_day", "avg_call_duration_sec", "avg_data_mb_per_day"]]
    clusters = df["cluster"]

    # Create user point
    user_point = np.array([[calls, duration, data_usage]])
    dists = euclidean_distances(user_point, features)
    nearest_idx = dists.argmin()
    user_cluster = clusters.iloc[nearest_idx]

    st.success(f"✅ Based on your input, you belong to **Cluster {user_cluster}**")
    st.info(f"Closest existing user: {df.iloc[nearest_idx]['user_id']}")

    # Prepare scatter data
    scatter_df = df.copy()
    scatter_df["is_new_user"] = False

    # Append new user row
    new_user_df = pd.DataFrame({
        "user_id": ["NEW_USER"],
        "calls_per_day": [calls],
        "avg_call_duration_sec": [duration],
        "avg_data_mb_per_day": [data_usage],
        "cluster": [user_cluster],
        "is_new_user": [True]
    })
    scatter_df = pd.concat([scatter_df, new_user_df], ignore_index=True)

    # Custom color assignment
    # Change: We'll now add a new user to its assigned cluster visually, and use a larger symbol
    # to highlight it. Plotly can't easily change symbol types based on data, but we can
    # give the new user its own 'cluster' and then use a custom symbol.
    
    # Recalculate plot to include new user
    fig = px.scatter(
        scatter_df,
        x=x_axis,
        y=y_axis,
        color="cluster", # Use the actual cluster label for color
        hover_data=["user_id", "is_new_user"],
        title=f"Clusters + Your Input (Belongs to Cluster {user_cluster})"
    )
    
    # Highlight the new user with a different symbol (e.g. star) and size
    fig.add_scatter(
        x=[calls], y=[duration],
        mode='markers',
        marker=dict(
            color='red',
            symbol='star',
            size=15,
            line=dict(width=2, color='DarkSlateGrey')
        ),
        name="NEW_USER"
    )

    st.plotly_chart(fig, use_container_width=True)