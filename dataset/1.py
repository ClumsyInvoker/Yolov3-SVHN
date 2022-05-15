import pickle
import numpy as np
import os
import h5py

svhnPath = '../Data'

subdir = 'train'

print('process folder : %s' % subdir)
filenames = []
dir = os.path.join(svhnPath, subdir)
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
