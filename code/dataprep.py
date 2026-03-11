import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def load_and_preprocess_uwb():
    # This finds the folder where dataprep.py is located (the 'code' folder)
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # This points to the sibling 'dataset' folder correctly
    dataset_dir = os.path.join(base_path, '..', 'dataset')
    
    all_frames = []
    
    print(f"Looking for files in: {os.path.abspath(dataset_dir)}")
    
    if not os.path.exists(dataset_dir):
        print(f"ERROR: The folder {dataset_dir} does not exist!")
        return None

    files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    
    if not files:
        print("ERROR: No CSV files found in the dataset folder!")
        return None

    for file in sorted(files):
        print(f"Loading {file}...")
        df = pd.read_csv(os.path.join(dataset_dir, file))
        all_frames.append(df)
    
    full_df = pd.concat(all_frames, ignore_index=True)

    # NEW: Strip whitespace from headers to avoid KeyErrors
    full_df.columns = full_df.columns.str.strip()
    
    # ... (rest of your CIR normalization code) ...
    return full_df

def rank_features(df):
    # Use the specific names provided in the dataset header/README [cite: 151-176]
    feature_names = [
        'Measured range (time of flight)', 'FP_IDX', 'FP_AMP1', 'FP_AMP2', 'FP_AMP3', 
        'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC', 'CH', 
        'FRAME_LEN', 'PREAM_LEN', 'BITRATE', 'PRFR'
    ]
    
    # Check which features actually exist in the dataframe to prevent errors
    existing_features = [f for f in feature_names if f in df.columns]
    
    X = df[existing_features]
    y = df['NLOS'] # Target: 1 for NLOS, 0 for LOS [cite: 151]

    print(f"Ranking {len(existing_features)} features...")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=existing_features)
    importances.sort_values().plot(kind='barh', color='skyblue')
    plt.title("Feature Importance: What predicts NLOS?")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_clean = load_and_preprocess_uwb()
    print(f"Data Prepared. Shape: {df_clean.shape}")
    
    # This fulfills your 'Feature Importance' and 'Visualization' requirements
    rank_features(df_clean)