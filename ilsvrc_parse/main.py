root_folder = '/data/efros/dataset/ILSVRC2012/'
data_filenames = 'val.txt'
data_folder = 'val/'

target_folder = 'sel_images/'
target_filenames = 'filenames.txt'

import numpy as np 
import os
import shutil

filenames = []
cls = []
print("Parsing filenames")
with open(root_folder + data_filenames, 'r') as f:
    for row in f.readlines():
	a, b = row.split()
        filenames.append(a)
        cls.append(int(b))
	
cls = np.array(cls, dtype=int)
filenames = np.array(filenames)
np.random.seed(1)

# select one image per class and copy it over to new location
ord=np.random.permutation(cls.size)
print(type(ord))
print(ord.dtype)
cls = cls[ord]
filenames = filenames[ord]
unq_cls, unq_ind = np.unique(cls, return_index=True)
print("Selecting one image per class from %d images of %d classes" %(cls.size, unq_cls.size))
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

with open(target_filenames, 'w') as f: 
    for i in range(len(unq_ind)):
        shutil.copyfile(root_folder + data_folder + filenames[unq_ind[i]], 
		        target_folder + filenames[unq_ind[i]]); 
        f.write("%s %d\n" % (filenames[unq_ind[i]], unq_cls[i])) 
