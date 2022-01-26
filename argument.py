import argparse

def get_args():
    parser = argparse.ArgumentParser(description="influence score based weighting")
    parser.add_argument('--gpu', required=True, default=3, type=int)
    parser.add_argument('--dataset', required=True, default='', choices=['adult', 'compas', 'bank', 'credit', 'celeba', 'utkface', 'retiring_adult'])
    parser.add_argument('--constraint', required=True, default='', choices=['dp', 'eo', 'eopp'])
    parser.add_argument('--method', required=True, default='', choices=['naive', 'influence', 'reweighting', 
                                                                        'naive_leave_k_out', 'naive_leave_bottom_k_out', 'leave_k_out_fine_tuning',
                                                                        'leave_random_k_out'])
    parser.add_argument('--epoch', required=True, default=0, type=int)
    parser.add_argument('--iteration', required=True, default=0, type=int)
    parser.add_argument('--scaler', default=None, type=float)
    parser.add_argument('--eta', default=None, type=float)
    parser.add_argument('--k', default=0, type=float)
    parser.add_argument('--term', default=1, type=int)
    parser.add_argument('--r', default = None, type=int)
    parser.add_argument('--t', default = None, type=int)
    parser.add_argument('--naive_acc', default=100, type=float)
    parser.add_argument('--naive_vio', default=100, type=float)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--idx_save', default=0, type=int)
    parser.add_argument('--model_save', default=0, type=int)
    parser.add_argument('--target', default=None)
    parser.add_argument('--sen_attr', default='sex')
    parser.add_argument('--fine_tuning', required=True, type=int)
    parser.add_argument('--main_option', required=True, choices=['fair_only', 'fair_only_fine_tuning', 'intersect', 'intersect_fine_tuning'])

    args = parser.parse_args()

    if args.seed is None:
        parser.error('requires --seed')

    if args.method == 'influence' and (args.scaler is None or args.r is None or args.t is None):
        parser.error('influence requires --scaler, --r, --t')
    if args.method == 'reweighting' and args.eta is None:
        parser.error('reweighting requires --eta')

    return args
