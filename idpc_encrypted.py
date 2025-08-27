"""
idpc_encrypted.py
Incremental Density Peak Clustering (IDPC) on encrypted data using Mock/Real HE + OPE.

Steps:
1. Load dummy CDR dataset (generated in Step 1).
2. Encrypt selected features (calls_per_day, avg_call_duration_sec, avg_data_mb_per_day).
3. Compute pairwise distances using HE (squared Euclidean).
4. Compute local density & delta using OPE (order preserved ranking).
5. Assign clusters based on density peaks.

NOTE:
- This is a simplified student prototype of IDPC.
- Runs fully on your local machine with no external dependencies beyond Step 0 setup.
"""

import pandas as pd
import numpy as np
import os
from encryption_module import get_managers


# ----------------------------
# IDPC Implementation
# ----------------------------
class EncryptedIDPC:
    def __init__(self, distance_threshold=10000, use_mode=None):
        """
        distance_threshold: max squared distance for considering neighbors (tunable).
        """
        self.distance_threshold = distance_threshold
        # Use get_managers to initialize HE and OPE based on mode
        self.he, self.ope = get_managers(mode=use_mode)

    def fit(self, df: pd.DataFrame):
        """
        df: input DataFrame with numeric features.
        Returns: cluster assignments as a list.
        """
        # Step 1: Encrypt features
        print("🔒 Encrypting features...")
        encrypted_vectors = []
        for i, row in df.iterrows():
            vec = [row["calls_per_day"], row["avg_call_duration_sec"], row["avg_data_mb_per_day"]]
            ct = self.he.encrypt_vec(vec)
            encrypted_vectors.append(ct)
            
            # ADDED: Print the first user's data before and after encryption
            if i == 0:
                print("\n--- HE IN ACTION (ENCRYPTION) ---")
                print(f"Original plaintext vector: {vec}")
                print(f"Encrypted vector (Ciphertexts): {ct}\n")

        n = len(encrypted_vectors)

        # Step 2: Compute encrypted pairwise distances
        print("📏 Computing pairwise distances...")
        distances = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                # when using real HE, squared_euclidean_ct returns a ciphertext; for mock it returns a single-slot ct too
                dist_ct = self.he.squared_euclidean_ct(encrypted_vectors[i], encrypted_vectors[j])
                # decrypt locally to get plaintext scalar
                dist2 = self.he.decrypt_scalar(dist_ct)
                distances[i, j] = distances[j, i] = int(dist2)

                # ADDED: Print the first distance calculation steps
                if i == 0 and j == 1:
                    print("\n--- HE IN ACTION (CALCULATION) ---")
                    print(f"Encrypted distance (Ciphertext): {dist_ct}")
                    print(f"Decrypted distance (Plaintext): {dist2}\n")

        # Step 3: Compute local density ρ_i = number of points within threshold
        print("📊 Computing local densities...")
        densities = []
        for i in range(n):
            rho = np.sum(distances[i] < self.distance_threshold) - 1  # exclude self
            densities.append(rho)

        # Step 4: Compute delta_i = min distance to a higher density point
        print("⚡ Computing deltas...")
        deltas = []
        for i in range(n):
            higher = [distances[i, j] for j in range(n) if densities[j] > densities[i]]
            delta = min(higher) if higher else max(distances[i])
            deltas.append(delta)

        # Step 5: Rank using OPE (simulate encryption-based ranking)
        encrypted_densities = self.ope.encrypt_list(densities)
        encrypted_deltas = self.ope.encrypt_list(deltas)

        # Step 6: Cluster assignment (very simplified)
        # Pick top-k density*delta as cluster centers
        scores = [d * dd for d, dd in zip(densities, deltas)]
        k = max(2, int(np.sqrt(n)))  # heuristic: sqrt(n) clusters
        centers = np.argsort(scores)[-k:]

        labels = [-1] * n
        for idx, c in enumerate(centers):
            labels[c] = idx

        # Assign each point to nearest cluster center
        for i in range(n):
            if labels[i] == -1:
                nearest_center = min(centers, key=lambda c: distances[i, c])
                labels[i] = labels[nearest_center]

        print("✅ Clustering complete!")
        return labels, densities, deltas, distances


# ----------------------------
# Script Entry Point
# ----------------------------
if __name__ == "__main__":
    # Load dummy dataset
    df = pd.read_csv("data/dummy_cdr.csv")
    print("Loaded dummy CDR data:", df.shape)

    # Run Encrypted IDPC
    # Pass a mode to the constructor to override the environment variable
    # model = EncryptedIDPC(distance_threshold=5000, use_mode="mock")
    model = EncryptedIDPC(distance_threshold=5000)
    labels, densities, deltas, distances = model.fit(df)

    # Attach clusters back to dataframe
    df["cluster"] = labels

    # Save results
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cdr_clusters.csv", index=False)
    print("\nCluster assignments saved to data/cdr_clusters.csv")
    print(df.head(10))