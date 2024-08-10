import os
import random
import shutil

def create_mock_dataset(source_dir, dest_dir, img_per_class, max_classes=None):
    os.makedirs(dest_dir, exist_ok=True)

    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    if max_classes:
        class_dirs = random.sample(class_dirs, min(max_classes, len(class_dirs)))

    for class_dir in class_dirs:
        source_cl_dir = os.path.join(source_dir, class_dir)
        dest_cl_dir = os.path.join(dest_dir, class_dir)

        os.makedirs(dest_cl_dir, exist_ok=True)

        image_files = [f for f in os.listdir(source_cl_dir) if f.lower().endswith('.jpg')]
        selected_img = random.sample(image_files, min(img_per_class, len(image_files)))

        for image in selected_img:
            source_path = os.path.join(source_cl_dir, image)
            dest_path = os.path.join(dest_cl_dir, image)
            shutil.copy2(source_path, dest_path)

def main():
    train_dir = '/content/drive/MyDrive/caltech_proj/data/Caltech256_Split/train'
    val_dir = '/content/drive/MyDrive/caltech_proj/data/Caltech256_Split/val'
    mock_train_dest = '/content/drive/MyDrive/caltech_proj/data/Caltech256_Split/mock_data/mock_train'
    mock_val_dest = '/content/drive/MyDrive/caltech_proj/data/Caltech256_Split/mock_data/mock_val'

    img_per_class_train = 4
    img_per_class_val = 2
    max_classes = 20

    print(f"Creating mock training dataset...")
    create_mock_dataset(train_dir, mock_train_dest, img_per_class_train, max_classes)

    print("\nCreating mock validation dataset...")
    create_mock_dataset(val_dir, mock_val_dest, img_per_class_val, max_classes)

    print("\nMock datasets created successfully!")
    print(f"Mock training data: {mock_train_dest}")
    print(f"Mock validation data: {mock_val_dest}")

if __name__ == '__main__':
    main()