import argparse

def get_args():
    parser = argparse.ArgumentParser(description="influence score based weighting")
    parser.add_argument('--dataset', required=True, default='', choices=['adult', 'compas', 'bank'])
    parser.add_argument('--constraint', required=True, default='', choices=['dp', 'eo', 'eopp'])
    parser.add_argument('--method', required=True, default='', choices=['naive', 'influence', 'reweighting', 'naive_leave_k_out', 'naive_leave_bottom_k_out', 'leave_k_out_fine_tuning'])
    parser.add_argument('--epoch', required=True, default=0, type=int)
    parser.add_argument('--iteration', required=True, default=0, type=int)
    parser.add_argument('--scaler', default=None, type=float)
    parser.add_argument('--eta', default=None, type=float)
    parser.add_argument('--k', default=0, type=int)
    parser.add_argument('--term', default=1, type=int)
    parser.add_argument('--r', default = 1, type=int)
    parser.add_argument('--t', default=5000, type=int)
    parser.add_argument('--naive_acc', default=100, type=float)
    parser.add_argument('--naive_vio', default=100, type=float)
    parser.add_argument('--seed', default=777, type=int)

    args = parser.parse_args()
    if args.method == 'influence' and args.scaler is None:
        parser.error('influence requires --scaler')
    if args.method == 'reweighting' and args.eta is None:
        parser.error('reweighting requires --eta')

    return args
