from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import PIL.Image as Image
from augument.siamese_transforms_with_mstar import SiameseTransform
import os
import csv
from config_metric_learning.config_mstar import configParams

class mastar_dataset(Dataset):
    def __init__(self, args, mode, transform, root_dir, num_per_class=10):
        self.mode = mode
        self.transform = transform
        self.root_dir = root_dir
        self.args = args
        self.num_per_class = num_per_class
        self.data_time = 6

        if self.mode == 'test':
           test_file_path = os.path.join(self.root_dir, 'test1.csv')
           image_paths, test_labels = self.read_csv(test_file_path)
           self.test_data, self.test_label = image_paths, test_labels

        elif self.mode == 'pre_train' or self.mode == 'classifier':
            train_file_path = os.path.join(self.root_dir, 'train1.csv')
            image_paths, train_labels = self.read_csv(train_file_path)
            size = len(train_labels)
            ind = list(range(size))
            random.shuffle(ind)
            image_paths = image_paths[ind]
            train_labels = train_labels[ind]
            # print(train_labels)
            self.train_data = image_paths
            self.train_label = train_labels

            if self.mode == 'pre_train':
                self.img0, self.img1, self.label = self.create_iterator(self.train_data, self.train_label, self.num_per_class)
                train_size = len(self.label)
                # print(train_size)
                indeies = list(range(train_size))
                random.shuffle(indeies)
                self.img0 = self.img0[indeies]
                self.img1 = self.img1[indeies]
                self.label = self.label[indeies]
            else:
                self.data, self.labels = self.create_iterator(self.train_data, self.train_label, self.num_per_class)
                train_size = len(self.labels)
                indeies = list(range(train_size))
                random.shuffle(indeies)
                self.data = self.data[indeies]
                self.labels = self.labels[indeies]

    def read_csv(self, file_path):
        label_name_dict = {}
        f = csv.reader(open(file_path, 'r'))

        label_index = 0
        label_names_ls = []
        labels_ls = []
        image_path_ls = []
        for i in f:
            image_path = i[0]
            label_name = i[1]
            image_path_ls.append(image_path)
            label_names_ls.append(label_name)

            if label_name not in label_name_dict.keys():
                label_name_dict[label_name] = label_index
                label_index += 1

        for i in range(len(label_names_ls)):
            label_name = label_names_ls[i]
            if label_name in label_name_dict.keys():
                labels_ls.append(label_name_dict[label_name])
            else:
                labels_ls.append('NAN')

        image_path_ls = np.array(image_path_ls)
        labels_ls = np.array(labels_ls)

        return image_path_ls, labels_ls

    def create_pairs(self, data, mastar_indices, num_per_class):
        x0_data_path = []
        x1_data_path = []
        label = []
        # n = min([len(mastar_indices[i]) for i in range(self.args.num_class)]) - 1
        # n = int(self.num_rate * n)
        print(f'num_per_class:{num_per_class},pre_train')
        for i in range(self.args.num_class):
            for j in range(num_per_class):
                z1, z2 = mastar_indices[i][j], mastar_indices[i][j+1]
                temp_z1 = [data[z1]] * self.data_time
                temp_z2 = [data[z2]] * self.data_time
                temp_label = [0] * self.data_time
                x0_data_path.extend(temp_z1)
                x1_data_path.extend(temp_z2)
                label.extend(temp_label)
                # x0_data_path.append(data[z1])
                # x1_data_path.append(data[z2])
                # label.append(0)

                for cnt in range(5):
                    k = i
                    while k == i:
                        inc = random.randint(0, self.args.num_class)
                        k = (k + inc) % self.args.num_class
                    z1, z2 = mastar_indices[i][j], mastar_indices[k][j]
                    temp_z1 = [data[z1]] * self.data_time
                    temp_z2 = [data[z2]] * self.data_time
                    temp_label = [1] * self.data_time
                    x0_data_path.extend(temp_z1)
                    x1_data_path.extend(temp_z2)
                    label.extend(temp_label)
                    # x0_data_path.append(data[z1])
                    # x1_data_path.append(data[z2])
                    # label.append(1)

        x0_data = np.array(x0_data_path)
        x1_data = np.array(x1_data_path)
        # print(x0_data.shape)
        # print(x0_data)
        label = np.array(label, dtype=np.float32)

        return x0_data, x1_data, label

    def create_classifier_data(self, data, mastar_indices, num_per_class):
        x_data_path = []
        labels = []
        # n = min([len(mastar_indices[i]) for i in range(self.args.num_class)])
        # n = int(n * self.num_rate)
        print(f'num_per_class:{num_per_class}, classifier_training')
        for i in range(self.args.num_class):
            for j in range(num_per_class):
                label = mastar_indices[i][j]
                temp_data = [data[label]] * self.data_time
                x_data_path.extend(temp_data)
                # x_data_path.append(data[label])
                temp_label = [i] * self.data_time
                labels.extend(temp_label)
                # labels.append(i)

        x_data = np.array(x_data_path)
        labels = np.array(labels, dtype=np.long)
        return x_data, labels

    def create_iterator(self, data, label, num_per_class):
        mastar_indices = [np.where(label == i)[0] for i in range(self.args.num_class)]
        # print(mastar_indices)
        if self.mode == 'pre_train':
            x0, x1, label = self.create_pairs(data, mastar_indices, num_per_class)
            return x0, x1, label
        else:
            data, labels = self.create_classifier_data(data, mastar_indices, num_per_class)
            return data, labels

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, label = self.test_data[index], self.test_label[index]
            img = Image.open(img_path)
            img = self.transform(img)
            return img, label
        elif self.mode == 'pre_train':
            img0_path, img1_path, label = self.img0[index], self.img1[index], self.label[index]
            img0 = Image.open(img0_path)
            img1 = Image.open(img1_path)
            img0, img1 = self.transform(img0, img1)
            return img0, img1, label
        else:
            img_path, label = self.data[index], self.labels[index]
            img = Image.open(img_path)
            img = self.transform(img)
            return img, label

    def __len__(self):
        if self.mode == 'test':
            return self.test_label.shape[0]
        elif self.mode == 'pre_train':
            return self.label.shape[0]
        else:
            return self.labels.shape[0]


class mastar_dataloader():
    def __init__(self, args, dataset, batchSize, num_works, data_dir, mode, num_per_class=10):
        self.dataset = dataset
        self.batchSize = batchSize
        self.num_works = num_works
        self.data_dir = data_dir
        self.mode = mode
        self.args = args
        self.num_per_class = num_per_class

        image_size = 32
        # degree = 60
        self.transform_test = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            # transforms.Resize(int(image_size * (8 / 7)), interpolation=Image.BICUBIC),  # 224 -> 256
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        self.transform_classifier_train = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.RandomRotation(degrees=(0, 360)),
            # transforms.RandomAffine(degree),
            # transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            transforms.Normalize((0.5,), (0.5,))
        ])

    def run(self):
        if self.mode == 'pre_train':
            transforms_train = SiameseTransform()
            train_dataset = mastar_dataset(args=self.args, root_dir=self.data_dir,
                                           mode='pre_train', transform=transforms_train,
                                           num_per_class=self.num_per_class)
            trainloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batchSize,
                # shuffle=True,
                num_workers=self.num_works,
                drop_last=True
            )
            return trainloader

        elif self.mode == 'test':
            test_dataset = mastar_dataset(args=self.args, root_dir=self.data_dir,
                                          mode='test', transform=self.transform_test)
            test_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batchSize,
                shuffle=False,
                num_workers=self.num_works,
                drop_last=True
            )
            return test_dataloader

        else:
            classifer_dataset = mastar_dataset(
                args=self.args, root_dir=self.data_dir,
                mode='classifier', transform=self.transform_classifier_train,
                num_per_class=self.num_per_class
            )
            trainloader = DataLoader(
                dataset=classifer_dataset,
                batch_size=self.batchSize,
                # shuffle=True,
                num_workers=self.num_works,
                drop_last=True
            )
            return trainloader


if __name__ == '__main__':
    root_dir = '../dataset/MSTAR_dataset/SARimage/SOC'
    args = configParams()

    args.pre_train_mode = 'pre_train'
    args.classifier_train_mode = 'classifier'
    args.eval_mode = 'test'
    args.num_rate = 1
    pre_train_dataloader = mastar_dataloader(
        args,
        args.dataset,
        args.train_batch_size, args.num_workers,
        root_dir,
        args.pre_train_mode,
        args.num_rate
    )
    pre_train_dataLoader = pre_train_dataloader.run()
    pre_train_dataloader = iter(pre_train_dataLoader)
    img0, img1, label = next(pre_train_dataloader)
    print(img0)













