import os
import pickle
from PIL import Image
import numpy as np

# Image.open
# cv2.load
# scipy.ndimage.imread


def makesubfolder(path):
    X = []
    y = []

    dirs = os.listdir(path)
    for class_id in range(len(dirs)):
        # Images to load
        images = os.listdir(os.path.join(path, dirs[class_id]))
        for im in images:
            loaded_im = np.asarray(Image.open(os.path.join(path, dirs[class_id], im)).resize((84, 84)).convert('RGB'))
            X.append(loaded_im)
            y.append(class_id)

    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y, dtype=np.int32)

    return X, y



basefolder = "datasets/miniimagenet"

print("Processing validation...")
val = makesubfolder(os.path.join(basefolder, "val"))

print("Processing train...")
train = makesubfolder(os.path.join(basefolder, "train"))

print("Processing test...")
test = makesubfolder(os.path.join(basefolder, "test"))

print("Saving...")
with open(os.path.join(basefolder,'miniimagenet.pkl'), 'wb') as f:
    pickle.dump({'train':train,'val':val,'test':test}, f, pickle.HIGHEST_PROTOCOL)


