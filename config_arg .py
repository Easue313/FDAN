import argparse

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--src_dataset', type=list, default=['Data1', 'Data2', 'Data3'])
    parser.add_argument('--tar_dataset', type=list, default=['Target data'])
    parser.add_argument('--data_length', type=int, default=4096)
    parser.add_argument('--points', type=int, default=350)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_size', type=int, default=400)
    parser.add_argument('--test_size', type=int, default=200)
    parser.add_argument('--sample_per_label', type=int, default=300)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--att_start_epoch', type=int, default=30)   #t1
    parser.add_argument('--aug_start_epoch', type=int, default=60)   #t2
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr1', type=float, default=0.0001)
    parser.add_argument('--src_num', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default="./log_files")
    parser.add_argument('--w_consistency', type=float, default=1)
    parser.add_argument('--rec_flag', type=bool, default=True)
    parser.add_argument('--w_rec', type=float, default=1)
    parser.add_argument('--w_ind', type=float, default=0.4)
    parser.add_argument('--w_aug', type=float, default=1)
    parser.add_argument('--w_mmd', type=float, default=0.3)
    parser.add_argument('--lamda', type=float, default=1)
    parser.add_argument('--mmd', type=bool, default=True)
    parser.add_argument('--mmd_sigma', type=float, default=1)
    args = parser.parse_args()
    return args
