# -*- coding: utf-8 -*-
import argparse
from CRNN_service import CRNN_Service

if __name__ == '__main__':

    # 1. Initialization arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute',
                        help='example:--execute train|predict|test')
    args = parser.parse_args()
    exe_type = args.execute
    types = ['train', 'predict', 'test']

    # 2. Perform the task.
    service = CRNN_Service()
    if exe_type == types[0]:
        service.train()
    elif exe_type == types[1]:
        print('result:', service.predict('./example/zuCa.png'))
    elif exe_type == types[2]:
        service.test()
    else:
        parser.print_help()
