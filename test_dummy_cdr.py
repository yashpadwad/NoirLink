import pandas as pd

def test_dummy_data(file_path="data/dummy_cdr.csv"):
    try:
        df = pd.read_csv(file_path)
        print("✅ Loaded dummy CDR data successfully!")
        print(f"Total records: {len(df)}")
        print("Sample rows:")
        print(df.head())
    except FileNotFoundError:
        print("❌ CSV file not found! Did you run dummy_cdr_generator.py first?")

if __name__ == "__main__":
    test_dummy_data()
