import argparse

def get_args():
    parser = argparse.ArgumentParser(description="get influence scores for images")
    parser.add_argument('--gpu', required=True, default=3, type=int)
    parser.add_argument('--dataset', required=True, default='', choices=['adult', 'compas', 'bank', 'celeba', 'utkface'])
    parser.add_argument('--sen_attr', default='sex')
    parser.add_argument('--target', required=True, default='')
    parser.add_argument('--constraint', required=True, default='', choices=['dp', 'eo', 'eopp'])
    parser.add_argument('--calc_option', required=True, default='', choices=['grad_V', 's_test', 'influence'])
    parser.add_argument('--seed', required=True, type=int)

    parser.add_argument('--r', default = None, type=int)
    parser.add_argument('--t', default = None, type=int)

    args = parser.parse_args()

    return args
