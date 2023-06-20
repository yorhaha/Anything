import os
import random
from os.path import join
from tqdm import tqdm
from shutil import copyfile

DATASET_PATH = './Dog_Cat/train'
TRAIN_SIZE = 9500
VALID_SIZE = 1000
TEST_SIZE = 2000
RADNOM_SEED = 3847
TARGET_PATH = './data'
random.seed(RADNOM_SEED)

for folder in ['train/dogs', 'validation/dogs', 'test/dogs', 'train/cats', 'validation/cats', 'test/cats']:
    os.makedirs(join(TARGET_PATH, folder), exist_ok=True)

def get_image_name_list():
    image_name_list = []
    for file_name in os.listdir(DATASET_PATH):
        if file_name.endswith('.jpg'):
            image_name_list.append(file_name)
    return image_name_list

def split_dog_cat(image_name_list):
    dog_list, cat_list = [], []
    for image_name in image_name_list:
        if image_name.startswith('dog'):
            dog_list.append(image_name)
        else:
            cat_list.append(image_name)
    return dog_list, cat_list

def copy_file(image_name_list, target_path):
    for image_name in tqdm(image_name_list):
        source_file = join(DATASET_PATH, image_name).replace('\\', '/')
        target_file = join(TARGET_PATH, target_path, image_name).replace('\\', '/')
        copyfile(source_file, target_file)

def test():
    import matplotlib.pyplot as plt
    from PIL import Image
    for i, folder in enumerate(['train/dogs', 'train/cats', 'validation/dogs', 'validation/cats', 'test/dogs', 'test/cats']):
        path = join(TARGET_PATH, folder)
        files = os.listdir(path)
        plt.subplot(3, 2, i + 1)
        img_path = join(TARGET_PATH, folder, files[0])
        image = Image.open(img_path)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'{folder}: {len(files)}')
    plt.show()

def main():
    image_name_list = get_image_name_list()
    dog_list, cat_list = split_dog_cat(image_name_list)
    random.shuffle(dog_list)
    random.shuffle(cat_list)
    dog_train, dog_valid, dog_test = dog_list[:TRAIN_SIZE], dog_list[TRAIN_SIZE:TRAIN_SIZE + VALID_SIZE], dog_list[TRAIN_SIZE + VALID_SIZE:]
    cat_train, cat_valid, cat_test = cat_list[:TRAIN_SIZE], cat_list[TRAIN_SIZE:TRAIN_SIZE + VALID_SIZE], cat_list[TRAIN_SIZE + VALID_SIZE:]
    print('dog_train: ', len(dog_train), ', dog_valid: ', len(dog_valid), ', dog_test: ', len(dog_test))
    print('cat_train: ', len(cat_train), ', cat_valid: ', len(cat_valid), ', cat_test: ', len(cat_test))
    copy_file(dog_train, 'train/dogs')
    copy_file(dog_valid, 'validation/dogs')
    copy_file(dog_test, 'test/dogs')
    copy_file(cat_train, 'train/cats')
    copy_file(cat_valid, 'validation/cats')
    copy_file(cat_test, 'test/cats')

if __name__ == '__main__':
    if os.path.isdir('./data/test/dogs') and not os.listdir('./data/test/dogs'):
        main()
    test()