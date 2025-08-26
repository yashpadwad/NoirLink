import pandas as pd
import numpy as np
import os

def generate_dummy_cdr(num_users=100, output_path="data/dummy_cdr.csv"):
    """
    Generates synthetic Call Detail Records (CDRs).
    
    Each record represents a user with:
    - user_id: Unique identifier
    - calls_per_day: Average number of calls per day
    - avg_call_duration_sec: Average duration of calls (in seconds)
    - avg_data_mb_per_day: Average data usage per day (in MB)
    """

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate synthetic values
    user_ids = [f"user_{i+1}" for i in range(num_users)]
    calls_per_day = np.random.randint(1, 20, size=num_users)
    avg_call_duration_sec = np.random.randint(30, 600, size=num_users)   # 0.5–10 mins
    avg_data_mb_per_day = np.random.randint(50, 5000, size=num_users)   # 50MB–5GB

    # Create DataFrame
    df = pd.DataFrame({
        "user_id": user_ids,
        "calls_per_day": calls_per_day,
        "avg_call_duration_sec": avg_call_duration_sec,
        "avg_data_mb_per_day": avg_data_mb_per_day
    })

    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Dummy CDR dataset generated at: {output_path}")

    return df


if __name__ == "__main__":
    # Generate 100 dummy users by default
    df = generate_dummy_cdr(100)
    print(df.head())
