from models.Tensor import TensorModel
from pickle import load
import os

# tensor_path = 'Z:/picklejar/collections/CF/tensors'
# save_path = 'Z:/picklejar/col.pickle'

tensor_path = 'C:/picklejar/collections/CF/tensors'
save_path = 'C:/picklejar/col.pickle'

with open(save_path, 'rb') as picklefile:
    cqr = load(picklefile)
    testcollection = cqr[0]
    q = cqr[1]
    r = cqr[2]
N = TensorModel(testcollection, tensor_path)
# q = 'What are the effects of calcium on the physical properties of mucus from CF patients'.upper().split()
print(q)

N.fit()
print('Done')