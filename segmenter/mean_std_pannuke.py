import numpy as np

path = '../Fold_{}/images/images.npy'
fold_sizes = np.zeros(3)
mean = np.zeros((3,3))
var = np.zeros((3,3))
for i in range(3):
    print(f" Fold_{i+1}")
    images = np.load(path.format(i+1))
    fold_sizes[i] = images.shape[0]
    mean[:,i] = np.mean(images,axis=(0,1,2))
    var[:,i] = np.var(images,axis=(0,1,2))
    print(f"mean_{i+1}:{mean}")
mean = (mean * (fold_sizes/(fold_sizes.sum()))).sum(1)
var = (var * (fold_sizes/(fold_sizes.sum()))).sum(1)
print("mean: ",mean)
print("std: ", np.sqrt(var))
