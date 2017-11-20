import argparse
import random

from basic_classifier_service import BasicClassifierService


def _prepare_data():
    data_x = []
    data_y = []
    for _ in range(0, 128):
        x = random.randint(-1000, 1000)
        if x > 0:
            y = [1., 0.]
        else:
            y = [0., 1.]
        data_x.append(x)
        data_y.append(y)
    return data_x, data_y


if __name__ == '__main__':

    # 1. Initialization arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute',
                        help='example:--execute train|predict|test')
    args = parser.parse_args()
    exe_type = args.execute
    types = ['train', 'predict', 'test']

    # 2. Perform the task.
    service = BasicClassifierService(input_size=1, output_size=2)
    if exe_type == types[0]:
        train_x, train_y = _prepare_data()
        service.train(train_x=train_x, train_y=train_y)
    elif exe_type == types[1]:
        print(service.predict([8, -8, -100]))
    elif exe_type == types[2]:
        test_x, test_y = _prepare_data()
        print(
            service.test(test_x=3, test_y=[0., 1.]))
    else:
        parser.print_help()
