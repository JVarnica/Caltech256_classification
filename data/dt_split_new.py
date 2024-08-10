import os
import random
import shutil
from tqdm import tqdm

def dataset_split(root, output_root, train_ratio=0.7, val_ratio=0.1):
    train_dir = os.path.join(output_root, 'train')
    val_dir = os.path.join(output_root, 'val')
    test_dir = os.path.join(output_root, 'test')
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    class_indices = {}
    current_idx = 0
    train_count = 0
    val_count = 0
    test_count = 0

    # Get list of valid class directories
    class_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d != "257.clutter"]

    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_path = os.path.join(root, class_dir)

        if class_dir not in class_indices:
            class_indices[class_dir] = current_idx
            current_idx += 1

        # Create class directories in sets
        train_class_dir = os.path.join(train_dir, class_dir)
        val_class_dir = os.path.join(val_dir, class_dir)
        test_class_dir = os.path.join(test_dir, class_dir)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        images_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images_files)

        train_split = int(len(images_files) * train_ratio)
        val_split = int(len(images_files) * (train_ratio + val_ratio))
        train_files = images_files[:train_split]
        val_files = images_files[train_split:val_split]
        test_files = images_files[val_split:]

        # Copy files to train directory
        for f in train_files:
            src = os.path.join(class_path, f)
            dst = os.path.join(train_class_dir, f)
            shutil.copy2(src, dst)
            train_count += 1

        for f in val_files:
            src = os.path.join(class_path, f)
            dst = os.path.join(val_class_dir, f)
            shutil.copy2(src, dst)
            val_count += 1

        # Copy files to test directory
        for f in test_files:
            src = os.path.join(class_path, f)
            dst = os.path.join(test_class_dir, f)
            shutil.copy2(src, dst)
            test_count += 1

    return train_count, val_count, test_count

def main():
    root = '256_ObjectCategories'
    output_root = 'Caltech256_Split'
    train_ratio = 0.7
    val_ratio = 0.1

    print(f"Starting dataset split...")
    print(f"Source directory: {root}")
    print(f"Output directory: {output_root}")
    
    try:
        dataset_split(root=root, output_root=output_root, 
                      train_ratio=train_ratio, val_ratio=val_ratio)
        print("Dataset split completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()