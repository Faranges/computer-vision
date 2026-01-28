import os
import shutil
import random

# 1. SET YOUR PATHS
# Current path where the unzipped data is
dataset_path = 'data/chest_xray' 
# The new folder that will be created
new_dataset_path = 'data/chest_xray_reorganized'

def reorganize():
    if not os.path.exists(new_dataset_path):
        print(f"Creating new dataset at {new_dataset_path}...")
        for split in ['train', 'val', 'test']:
            for cls in ['NORMAL', 'PNEUMONIA']:
                os.makedirs(f'{new_dataset_path}/{split}/{cls}', exist_ok=True)

        for cls in ['NORMAL', 'PNEUMONIA']:
            all_files = []
            # Gather all images from existing train, val, and test
            for split in ['train', 'val', 'test']:
                source_folder = os.path.join(dataset_path, split, cls)
                if os.path.exists(source_folder):
                    files = [f for f in os.listdir(source_folder) if not f.startswith('.')]
                    all_files.extend([(file, source_folder) for file in files])

            # Shuffle to ensure random distribution
            random.seed(42) # Fixed seed so you get the same result every time
            random.shuffle(all_files)

            # Define split points (80% Train, 10% Val, 10% Test)
            train_idx = int(len(all_files) * 0.8)
            val_idx = int(len(all_files) * 0.9)

            train_files = all_files[:train_idx]
            val_files = all_files[train_idx:val_idx]
            test_files = all_files[val_idx:]

            # Copy files to new structure
            for file, source_folder in train_files:
                shutil.copy(os.path.join(source_folder, file), f'{new_dataset_path}/train/{cls}/{file}')

            for file, source_folder in val_files:
                shutil.copy(os.path.join(source_folder, file), f'{new_dataset_path}/val/{cls}/{file}')

            for file, source_folder in test_files:
                shutil.copy(os.path.join(source_folder, file), f'{new_dataset_path}/test/{cls}/{file}')
        
        print("Success! Data reorganized.")
    else:
        print("Reorganized folder already exists. Skipping.")

if __name__ == "__main__":
    reorganize()