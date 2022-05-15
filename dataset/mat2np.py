import pickle
import numpy as np
import os
import h5py

svhnPath = '../Data'


def loadSvhn(path, subdir):
    print('process folder : %s' % subdir)
    filenames = []
    dir = os.path.join(path, subdir)
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


if __name__ == '__main__':
    for sub_dir in ['train', 'test']:
        data_sets = loadSvhn(svhnPath, sub_dir)
        # data_sets = [{'dir': './', 'file': ('01', '.png'),
        #              'boxes': {'label': ['0'], 'left': [12], 'top': [10], 'width': [20], 'height': [30]}}]
        print('processing locations to txt file ...')
        for ds in data_sets:
            txt_file = os.path.join(ds['dir'], ds['file'][0] + '.txt')
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
                    lines.append('%s,%s,%s,%s,%s' % (int(label), left, top, width, height))
                fs.write('\n'.join(lines))
        filename = 'all.dat'
        with open(os.path.join(svhnPath, sub_dir, filename), 'wb') as f:
            pickle.dump(data_sets, f)

        print(sub_dir, ' done.')


