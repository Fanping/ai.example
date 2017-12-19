# -*- coding: utf-8 -*-
import argparse
from text_classify_cnn_service import TextClassifierCNNService

if __name__ == '__main__':

    # 1. Initialization arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute',
                        help='example:--execute train|predict|test')
    args = parser.parse_args()
    exe_type = args.execute
    types = ['train', 'predict', 'test']

    # 2. Perform the task.
    service = TextClassifierCNNService()
    if exe_type == types[0]:
        service.train()
    elif exe_type == types[1]:
        content = '考生必看：09年6月六级考试写作小贴士一、从评分标准来看对策1.CET作文题采用总体评分'
        print(service.predict(content))
    elif exe_type == types[2]:
        service.test()
    else:
        parser.print_help()
