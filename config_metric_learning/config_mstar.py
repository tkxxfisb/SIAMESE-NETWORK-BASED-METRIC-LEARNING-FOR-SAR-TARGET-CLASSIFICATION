import argparse

def configParams():
    parser = argparse.ArgumentParser(description='Pytorch mstar Training')
    parser.add_argument('--dataset', type=str, default='mstar')
    parser.add_argument('--data_path', type=str, default='./dataset/MSTAR_dataset/SARimage/SOC')
    parser.add_argument('--train_batch_size', type=int, default=20)
    parser.add_argument('--train_number_epochs', type=int, default=300)
    parser.add_argument('--save_img', type=str, default='./img')
    parser.add_argument('--save_plot', type=str, default='./plot')
    parser.add_argument('--inplane', type=int, default=1)
    parser.add_argument('--gpu_nums', type=int, default=3)

    parser.add_argument('--gpu_id', type=str, default='0,2,3')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--root', type=str, default='./checkpoint')
    parser.add_argument('--classifier_train_epochs', type=int, default=300)
    parser.add_argument('--classifier_batch_size', type=int, default=20)

    # SGD optimizer parameters
    parser.add_argument('--base_lr', type=float, default=0.03)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_lr', type=float, default=0.)
    parser.add_argument('--final_lr', type=float, default=0.)

    # sgd optimizer parameters training
    parser.add_argument('--a_base_lr', type=float, default=0.03)
    parser.add_argument('--a_num_workers', type=int, default=4)
    parser.add_argument('--a_momentum', type=float, default=0.9)
    parser.add_argument('--a_weight_decay', type=float, default=0.0005)
    parser.add_argument('--a_warmup_epochs', type=int, default=10)
    parser.add_argument('--a_warmup_lr', type=float, default=0.)
    parser.add_argument('--a_final_lr', type=float, default=0.)

    # Adam optimizer
    parser.add_argument('--base_lr_adam', type=float, default=1e-3)
    parser.add_argument('--warmup_lr_adam', type=float, default=0.)
    parser.add_argument('--warmup_epochs_for_adam', type=int, default=10)
    parser.add_argument('--epochs_for_weight_decay_adam', type=int, default=50)
    parser.add_argument('--lr_rate_adam', type=float, default=0.1)
    args = parser.parse_args()
    return args