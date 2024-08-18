import tarfile
import pickle
import numpy as np 
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_image(array, path):
    img = Image.fromarray(array)
    img.save(path)

def process_cifar100(data_dir, output_dir):
    
    #meta file to get class names. the file is just list of class
    meta = unpickle(os.path.join(data_dir, 'meta'))
    fine_label_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]

    # Get images/labels
    train_data = unpickle(os.path.join(data_dir, 'train'))
    t_images = train_data[b'data']
    t_labels = train_data[b'fine_labels']

    train_images, val_images, train_labels, val_labels = train_test_split(
        t_images, t_labels, test_size=0.125, random_state=42) # 0.125 of 80= 10% overall
    
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        class_name = fine_label_names[label]
        class_dir = os.path.join(output_dir, 'train', class_name)
        os.makedirs(class_dir, exist_ok=True)
        filename = f"{i:06d}.png"
        save_image(image, os.path.join(class_dir, filename))

    for i, (image, label) in enumerate(zip(val_images, val_labels)):
        image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        class_name = fine_label_names[label]
        class_dir = os.path.join(output_dir, 'val', class_name)
        os.makedirs(class_dir, exist_ok=True)
        filename = f"{i:06d}.png"
        save_image(image, os.path.join(class_dir, filename))

    test_data = unpickle(os.path.join(data_dir, 'test'))
    for i, (image, label) in enumerate(zip(test_data[b'data'], test_data[b'fine_labels'])):
        image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        class_name = fine_label_names[label]
        class_dir = os.path.join(output_dir, 'test', class_name)
        os.makedirs(class_dir, exist_ok=True)
        filename = f"{i:06d}.png"
        save_image(image, os.path.join(class_dir, filename))

    print("Extraction complete")

def main(): 
    data_dir = '/content/drive/MyDrive/cifar100/cifar-100-python'
    output_dir = '/content/drive/MyDrive/caltech_proj/data/Cifar100'
    process_cifar100(data_dir, output_dir)

if __name__ == '__main__':
    main()
