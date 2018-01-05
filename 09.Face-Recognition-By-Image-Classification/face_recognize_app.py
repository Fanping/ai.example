# -*- coding: utf-8 -*-
import argparse
from face_recognize_service import FaceRecognizeService

if __name__ == '__main__':

    # 1. Initialization arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute',
                        help='example:--execute train|test')
    args = parser.parse_args()
    exe_type = args.execute
    types = ['train', 'test']

    # 2. Perform the task.
    service = FaceRecognizeService()
    if exe_type == types[0]:
        service.train()
    elif exe_type == types[1]:
        service.test()
    else:
        parser.print_help()
