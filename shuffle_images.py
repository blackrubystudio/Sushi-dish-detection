"""
Shuffle and renames images in the given directory.

Usage:
python3 shuffle_images.py --dataset ${PWD}/path/to/data
"""
import os
import random

def suffle_images (path):
    filenames = os.listdir(path)
    random.shuffle(filenames)

    for index, filename in enumerate(filenames):
        if '.jpg' in filename:
            os.rename(os.path.join(path, filename), 
                      os.path.join(path, 'dish_{}.jpg'.format(index)))

if __name__ == "__main__":
    # Parse command line argument
    import argparse
    parser = argparse.ArgumentParser(description='Shuffle images for train deep learning models')
    parser.add_argument('--dataset', required=False,
                        default=os.getcwd(),
                        metavar='/path/to/dataset',
                        help='Directory of the dataset')
    arg = parser.parse_args()

    print('Shuffle images start in {}...'.format(arg.dataset))
    suffle_images(arg.dataset)
    print('Finish!')
