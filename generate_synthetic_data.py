import numpy as np
import pandas as pd
import os

def generate_synthetic_logs(num_samples=5000):
    np.random.seed(42)
    print("Generating synthetic well-logging data...")
    
    # Simulate 5 logging curves (RT, AC, SP, GR, DEN)
    rt = np.random.lognormal(mean=1.0, sigma=0.5, size=num_samples)
    ac = np.random.normal(loc=250, scale=40, size=num_samples)
    sp = np.random.normal(loc=-20, scale=15, size=num_samples)
    gr = np.random.normal(loc=80, scale=30, size=num_samples)
    den = np.random.normal(loc=2.4, scale=0.2, size=num_samples)
    
    # Simulate 7 lithologic labels (0-6)
    labels = np.random.randint(0, 7, size=num_samples)
    
    # organized into DataFrame
    data = pd.DataFrame({'RT': rt, 'AC': ac, 'SP': sp, 'GR': gr, 'DEN': den, 'Lithology': labels})
    
    # Saved as CSV
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/synthetic_logs.csv', index=False)
    print("Data saved to data/synthetic_logs.csv")

if __name__ == "__main__":
    generate_synthetic_logs()