import os
import csv
import codecs

root = './'
root1 = './dataset/MSTAR_dataset/SARimage/SOC'
train_file_path = os.path.join(root, 'train.csv')
test_file_path = os.path.join(root, 'test.csv')
for file_name in [train_file_path, test_file_path]:
    f = csv.reader(open(file_name, 'r'))
    ls = []
    for i in f:
        ls.append(i)

    print(ls[:2])
    for temp_ls in ls:
        image_path = temp_ls[0]
        image_path = image_path.strip().split('/')
        new_image_path = '/'.join(image_path[4:])
        # new_image_path = './' + new_image_path
        # new_image_path = os.path.join(root1, new_image_path)
        new_image_path = root1 + '/' + new_image_path
        # print(new_image_path)
        temp_ls[0] = new_image_path
        # break

    print(ls[:2])
    new_train_file_path = None
    new_test_file_path = None
    if file_name == train_file_path:
        new_train_file_path = os.path.join(root, 'train1.csv')
    else:
        new_test_file_path = os.path.join(root, 'test1.csv')

    if new_train_file_path is not None:
        f = codecs.open(new_train_file_path, 'w', 'utf-8')
    else:
        f = codecs.open(new_test_file_path, 'w', 'utf-8')
    writer = csv.writer(f)
    for i in ls:
        writer.writerow(i)
    f.close()


