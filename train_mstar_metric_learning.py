from Dataloader.dataLoader_mastar_metric_learning import mastar_dataloader
from network.resnet_metric_learning import *
import torch.optim as optim
from plt import show_plot, show_acc_plot
from PIL import Image
from config_metric_learning.config_mstar import configParams
from utils import Average, LR_scheduler_for_mstar
import random
import matplotlib as mpl
import os
from shutil import copyfile, rmtree
import time
mpl.use('Agg')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(state, epoch, model_name='pre_model'):
    checkpoint_dir = './checkpoint/metric_learning'
    model_path = os.path.join(checkpoint_dir, model_name)
    train_dir = os.path.join(model_path, 'train')

    src_model_path = os.path.join(train_dir, f'{epoch}-train.pth')
    torch.save(state, src_model_path)

    is_best_dir = os.path.join(model_path, 'isbest')
    if not os.path.exists(is_best_dir):
        os.makedirs(is_best_dir)

    target_model_path = os.path.join(is_best_dir, 'is_best.pth')
    copyfile(src_model_path, target_model_path)


def clear_dirs(model_name='pre_model'):
    checkpoint_dir = './checkpoint/metric_learning'
    train_dir = os.path.join(checkpoint_dir, model_name, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    else:
        rmtree(train_dir)
        os.makedirs(train_dir)


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main(is_pre_train=True, is_train=False):
    args = configParams()
    args.pre_train_mode = 'pre_train'
    args.classifier_train_mode = 'classifier'
    args.eval_mode = 'test'
    args.num_per_class = 20
    # torch.cuda.set_device(3)
    set_seed(args)

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    gpu_nums = len(args.gpu_id.split(','))

    if is_pre_train:
        pre_train_dataLoader = mastar_dataloader(
            args,
            args.dataset,
            args.train_batch_size, args.num_workers,
            args.data_path, args.pre_train_mode,
            args.num_per_class
        )
        pre_train_dataLoader = pre_train_dataLoader.run()
        pre_model = ResNet34(args.inplane).to(device)
        # pre_model = ResNet34(args.inplane)
        # if gpu_nums > 1:
        #     pre_model = nn.DataParallel(pre_model, device_ids=[0, 1, 2]).to(device)
        # pre_optimizer = optim.SGD(pre_model.parameters(),
        #                           lr=args.base_lr * args.train_batch_size / 256, momentum=args.momentum,
        #                           weight_decay=args.weight_decay)
        #
        # pre_lr_scheduler = LR_Scheduler(
        #     pre_optimizer,
        #     warmup_epochs=args.warmup_epochs,
        #     warmup_lr=args.warmup_lr * args.train_batch_size / 256,
        #     num_epochs=args.train_number_epochs,
        #     base_lr=args.base_lr * args.train_batch_size / 256,
        #     final_lr=args.final_lr * args.train_batch_size / 256,
        #     iter_per_epoch=len(pre_train_dataLoader),
        #     constant_predictor_lr=True
        # )
        pre_optimizer = optim.Adam(pre_model.parameters(), lr=args.base_lr)
        pre_lr_scheduler = LR_scheduler_for_mstar(
            pre_optimizer,
            base_lr=args.base_lr_adam,
            warmup_lr=args.warmup_lr_adam,
            num_epochs=args.train_number_epochs,
            warmup_epochs=args.warmup_epochs_for_adam,
            iter_per_epoch=len(pre_train_dataLoader),
            lr_decay_rate=args.lr_rate_adam
        )


        counter, loss_history = pre_train(args, pre_model, pre_optimizer, pre_train_dataLoader, pre_lr_scheduler)
        show_plot(args, counter, loss_history, dir_name='metric_learning/pre_training')

    else:
        if is_train:
            train_dataLoader = mastar_dataloader(
                args,
                args.dataset,
                args.train_batch_size,
                args.num_workers,
                args.data_path,
                args.classifier_train_mode,
                args.num_per_class
            )
            train_dataLoader = train_dataLoader.run()
            criterion = nn.CrossEntropyLoss()

            pre_model = ResNet34(args.inplane).to(device)
            root = args.root
            pre_model_path = 'metric_learning/pre_model'
            model_path = os.path.join(root, pre_model_path, 'isbest', 'is_best.pth')
            checkpoint = torch.load(model_path)

            pre_model.load_state_dict(checkpoint['network'])
            # for param in pre_model.parameters():
            #     param.requires_grad = False
            in_features = checkpoint['output_dim']
            # print(in_features)
            model = MLP_classifier(in_features).to(device)
            # model = nn.DataParallel(model, device_ids=[0, 1, 2])

            # optimizer = optim.SGD(
            #     model.parameters(),
            #     lr=args.base_lr * args.train_batch_size / 256, momentum=args.momentum,
            #     weight_decay=args.weight_decay
            # )
            #
            # lr_scheduler = LR_Scheduler(
            #     optimizer,
            #     warmup_epochs=args.warmup_epochs,
            #     warmup_lr=args.warmup_lr * args.train_batch_size / 256,
            #     num_epochs=args.train_number_epochs,
            #     base_lr=args.base_lr * args.train_batch_size / 256,
            #     final_lr=args.final_lr * args.train_batch_size / 256,
            #     iter_per_epoch=len(train_dataLoader),
            #     constant_predictor_lr=True
            # )
            optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
            lr_scheduler = LR_scheduler_for_mstar(
                optimizer,
                base_lr=args.base_lr_adam,
                warmup_lr=args.warmup_lr_adam,
                num_epochs=args.train_number_epochs,
                warmup_epochs=args.warmup_epochs_for_adam,
                iter_per_epoch=len(train_dataLoader),
                lr_decay_rate=args.lr_rate_adam
            )
            counter, loss_history = train_classifier(args, pre_model, model, optimizer, criterion, lr_scheduler, train_dataLoader)
            show_plot(args, counter, loss_history, dir_name='metric_learning/training')
        else:
            test_dataLoader = mastar_dataloader(
                args,
                args.dataset,
                args.train_batch_size,
                args.num_workers,
                args.data_path,
                args.eval_mode
            )
            test_dataLoader = test_dataLoader.run()
            eval_classifier(args, test_dataLoader)


def pre_train(args, model, optimizer, train_dataLoader, scheduler):
    counter = []
    loss_history = []
    iteration_number = 0
    train_loss = Average()
    clear_dirs()
    for epoch in range(0, args.train_number_epochs):
        model.train()
        for idx, (img0, img1, labels) in enumerate(train_dataLoader):
            # print('img0.shape', img0.shape)
            bs = labels.shape[0]
            optimizer.zero_grad()
            img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)
            out = model(img0, img1)
            loss = F.binary_cross_entropy_with_logits(out, labels)
            train_loss.update(loss.item(), bs)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if idx % 10 == 0:
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(train_loss.avg)
        del img0, img1, labels
        print(f'Epoch number:{epoch}, current_loss:{train_loss.avg}')
        state = {}
        state['network'] = model.state_dict()
        state['optimizer'] = optimizer.state_dict()
        state['epoch'] = epoch
        state['output_dim'] = model.img_output_dim
        save_checkpoint(state, epoch)
    return counter, loss_history


def train_classifier(args, pre_model, model, optimizer, criterion, scheduler, train_dataLoader):

    train_loss = Average()

    counter = []
    loss_history = []
    iteration_num = 0
    clear_dirs(model_name='classifier')
    for epoch in range(0, args.classifier_train_epochs):
        pre_model.eval()
        model.train()
        for idx, (imgs, labels) in enumerate(train_dataLoader):
            bs = labels.shape[0]
            optimizer.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feature_vector = pre_model.extract_feature(imgs)
                # print(feature_vector.shape)
            out = model(feature_vector)
            loss = criterion(out, labels)
            train_loss.update(loss.item(), bs)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if idx % 10 == 0:
                iteration_num += 10
                counter.append(iteration_num)
                loss_history.append(train_loss.avg)
        del imgs, labels
        print(f'Epoch number:{epoch}, current_loss:{train_loss.avg}')
        state = {'classifier': model.state_dict(), 'epoch': epoch}
        save_checkpoint(state, epoch, model_name='classifier')
    return counter, loss_history


def eval_classifier(args, eval_dataLoader):
    pre_model = ResNet34(args.inplane).to(device)
    root = args.root
    pre_model_path = 'metric_learning/pre_model'
    model_path = os.path.join(root, pre_model_path, 'isbest', 'is_best.pth')
    checkpoint = torch.load(model_path)
    pre_model.load_state_dict(checkpoint['network'])
    pre_model.eval()

    in_features = checkpoint['output_dim']
    classifier = MLP_classifier(in_features).to(device)

    best_acc = 0.
    best_epoch = 0
    for epoch in range(args.classifier_train_epochs):
        classifier_path = 'metric_learning/classifier'
        classifier_path = os.path.join(root, classifier_path, 'train')
        classifier_path = os.path.join(classifier_path, f'{epoch}-train.pth')
        checkpoint = torch.load(classifier_path)
        classifier.load_state_dict(checkpoint['classifier'])

        classifier.eval()
        correct = 0.
        total_nums = len(eval_dataLoader.dataset)
        with torch.no_grad():
            for idx, (imgs, labels) in enumerate(eval_dataLoader):
                imgs, labels = imgs.to(device), labels.to(device)
                features = pre_model.extract_feature(imgs)
                prob = classifier(features)
                pred = torch.argmax(prob, dim=1)
                correct += torch.eq(pred, labels.data.view_as(pred)).sum()
                correct = correct.item()
            acc = round(100. * correct / total_nums, 3)
            print(f'correct:{correct}, total_nums:{total_nums}, accuracy:{acc}')
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch

    print(f'best_acc:{best_acc}, best_epoch:{best_epoch}')


def pre_train_main():
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('start_train_time:', start_time)
    main(is_pre_train=True)
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('-----------------------------------------')
    print('start_train_time:', start_time)
    print('end_train_time:', end_time)
    print('-----------------------------------------')


def train_main():
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('start_train_time:', start_time)
    main(is_pre_train=False, is_train=True)
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('-----------------------------------------')
    print('start_train_time:', start_time)
    print('end_train_time:', end_time)
    print('-----------------------------------------')


def eval_main():
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('start_test_time:', start_time)
    main(is_pre_train=False, is_train=False)
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('-----------------------------------------')
    print('start_train_time:', start_time)
    print('end_train_time:', end_time)
    print('-----------------------------------------')


if __name__ == '__main__':
    pre_train_main()
    train_main()
    eval_main()


