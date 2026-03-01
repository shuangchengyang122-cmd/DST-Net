import numpy as np
import pandas as pd
from dst_net import build_dst_net
import os

# --- Hyperparameters ---
WINDOW_SIZE = 12  # 1.2m depth window at 0.1m resolution
NUM_FEATURES = 5  # RT, AC, SP, GR, DEN
NUM_CLASSES = 7   # 7 lithology types
EPOCHS = 10
BATCH_SIZE = 64

def create_sliding_windows(data, labels, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

def main():
    # 1. Load Data
    data_path = 'data/synthetic_logs.csv'
    if not os.path.exists(data_path):
        print("Error: Synthetic data not found. Please run 'generate_synthetic_data.py' first.")
        return
        
    df = pd.read_csv(data_path)
    features = df[['RT', 'AC', 'SP', 'GR', 'DEN']].values
    labels = df['Lithology'].values
    
    # 2. Preprocessing (Sliding Window)
    print("Applying sliding window...")
    X, y = create_sliding_windows(features, labels, WINDOW_SIZE)
    
    # Train/Test Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 3. Build Model
    model = build_dst_net(WINDOW_SIZE, NUM_FEATURES, NUM_CLASSES)
    model.summary()
    
    # 4. Train Model
    print("\nStarting DST-Net training...")
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)
    
    # 5. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Evaluation Result] Accuracy on synthetic test set: {accuracy*100:.2f}%")
    print("Note: This performance is based on synthetic random data for peer review validation purposes.")

if __name__ == "__main__":
    main()