import numpy as np

path = '../Fold_{}/images/images.npy'
fold_sizes = np.zeros((3,2))
mean = np.zeros((3,3))
var = np.zeros((3,3))
for i in range(3):
    print(f" Fold_{i+1}")
    images = np.load(path.format(i+1))
    fold_sizes[i,0] = images.shape[0]
    fold_sizes[i-1,0] = images.shape[0]
    mean[:,i] = np.mean(images,axis=(0,1,2))
    var[:,i] = np.var(images,axis=(0,1,2))
mean = (mean * (1/fold_sizes.sum(1))).sum(1)
var = (var * (1/fold_sizes.sum(1))).sum(1)
print("mean: ",mean)
print("std: ", np.sqrt(var))