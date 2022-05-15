import pickle
import numpy as np
import os
import h5py
import random

svhnPath = './Data/SVHN/'
imagesPath = './Data/SVHN/images/' # 存放图片
labelsPath = './Data/SVHN/labels/' # 存放标签和bbox
sub_dirs = ['train', 'test']


def load_svhn(path, subdir):
    print('process folder : %s' % subdir)
    filenames = []
    dir = os.path.join(path, subdir)
    dir = dir.replace("\\", "/")
    for filename in os.listdir(dir):
        filenameParts = os.path.splitext(filename)
        if filenameParts[1] != '.png':
            continue
        filenames.append(filenameParts)
    svhnMat = h5py.File(name=os.path.join(dir, 'digitStruct.mat'), mode='r')
    datasets = []
    filecounts = len(filenames)
    for idx, file in enumerate(filenames):
        boxes = {}
        filenameNum = file[0]
        item = svhnMat['digitStruct']['bbox'][int(filenameNum) - 1].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = svhnMat[item][key]
            values = [svhnMat[attr[i].item()][0][0]
                      for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
            boxes[key] = values
        datasets.append({'dir': dir, 'file': file, 'boxes': boxes})
        if idx % 10 == 0:
            print('-- loading %d / %d' % (idx, filecounts))

    return datasets


def save_labels():
    os.makedirs(labelsPath, exist_ok=True)

    for sub_dir in sub_dirs:
        data_sets = load_svhn(imagesPath, sub_dir)

        # data_sets = [{'dir': './Data/SVHN/images/train', 'file': ('1', '.png'),
        #              'boxes': {'label': ['0'], 'left': [12], 'top': [10], 'width': [20], 'height': [30]}}]

        print('processing locations to txt file ...')
        txt_dir = os.path.join(labelsPath, sub_dir)
        txt_dir = txt_dir.replace("\\", "/")
        os.makedirs(txt_dir, exist_ok=True)

        for ds in data_sets:
            txt_file = os.path.join(txt_dir, ds['file'][0] + '.txt')

            boxes = ds['boxes']
            labels = boxes['label']
            lines = []
            with open(txt_file, mode='w', encoding='utf-8') as fs:
                for i in range(len(labels)):
                    label = boxes['label'][i]
                    left = boxes['left'][i]
                    top = boxes['top'][i]
                    width = boxes['width'][i]
                    height = boxes['height'][i]
                    lines.append('%s %s %s %s %s' % (int(label) % 10, left, top, width, height))
                fs.write('\n'.join(lines))

        print(sub_dir, ' done.')


def split_train_and_valid(data, ratio=0.2):
    valid_data = random.sample(data, int(ratio*len(data)))
    train_data = [x for x in data if x not in valid_data]
    return train_data, valid_data


def get_train_valid_test():
    for sub_dir in sub_dirs:
        dir = os.path.join(imagesPath, sub_dir)
        dir = dir.replace("\\", "/")
        filenames = []
        for filename in os.listdir(dir):
            filenameParts = os.path.splitext(filename)
            if filenameParts[1] != '.png':
                continue
            filenames.append(os.path.relpath(os.path.join(dir, filename)).replace("\\", '/'))

        if sub_dir == 'train':
            train, valid = split_train_and_valid(filenames, ratio=0.2)
            train_txt = os.path.join(svhnPath, 'train.txt')
            with open(train_txt, mode='w', encoding='utf-8') as fs:
                for filename in train:
                    fs.write(filename+'\n')

            valid_txt = os.path.join(svhnPath, 'valid.txt')
            with open(valid_txt, mode='w', encoding='utf-8') as fs:
                for filename in valid:
                    fs.write(filename+'\n')

        if sub_dir == 'test':
            test = filenames
            test_txt = os.path.join(svhnPath, 'test.txt')
            with open(test_txt, mode='w', encoding='utf-8') as fs:
                for filename in test:
                    fs.write(filename+'\n')

    print("Split Done.")


if __name__ == '__main__':
    get_train_valid_test()

    save_labels()


