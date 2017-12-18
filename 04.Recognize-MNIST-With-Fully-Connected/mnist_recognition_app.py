import argparse
from PIL import Image
import numpy as np
from mnist_recognition_service import MnistRegService


def _normalize(file_path):
    Im = Image.open(file_path)

    pixel = Im.load()

    width, height = Im.size
    data_image = []
    for x in range(0, width):
        for y in range(0, height):
            data_image.append(pixel[y, x])
    return data_image


if __name__ == '__main__':

    # 1. Initialization arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute',
                        help='example:--execute train|predict|test')
    args = parser.parse_args()
    exe_type = args.execute
    types = ['train', 'predict', 'test']

    # 2. Perform the task.
    service = MnistRegService()
    if exe_type == types[0]:
        service.train()
    elif exe_type == types[1]:
        nums = service.predict([_normalize('../data/mnist/visual/mnist_9.jpg'),
                                _normalize('../data/mnist/visual/mnist_5.jpg')])
        print (nums)
    elif exe_type == types[2]:
        service.test()
    else:
        parser.print_help()
