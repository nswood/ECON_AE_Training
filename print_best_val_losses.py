import os
import pandas as pd
import argparse

def print_min_val_loss(base_dir):
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            csv_path = os.path.join(subdir_path, 'df.csv')
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                if 'val_loss' in df.columns:
                    min_val_loss = df['val_loss'].min()
                    print(f"Min val_loss for {subdir}: {min_val_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print minimum validation loss from CSV files in subdirectories.")
    parser.add_argument('--base_dir', type=str, help='The base directory path')
    args = parser.parse_args()
    
    print_min_val_loss(args.base_dir)