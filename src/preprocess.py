import pandas as pd
import os

def preprocess(input, output):
    data = pd.read_csv(input)
    print("Columns before dropping:", data.columns.tolist())  # Debugging

    if 'Id' in data.columns:
        data = data.drop(columns=['Id'])
    else:
        print("'Id' column not found. No columns were dropped.")

    print("Columns after dropping:", data.columns.tolist())  # Debugging

    os.makedirs(os.path.dirname(output), exist_ok=True)
    data.to_csv(output, header=True, index=False)
    print(f"Preprocessed data saved to {output}")

if __name__ == "__main__":
    input = r"C:\MLOPS\wine_quality-Aws-\data\WineQT.csv"
    output = r"C:\MLOPS\wine_quality-Aws-\data\data.csv"
    preprocess(input, output)
