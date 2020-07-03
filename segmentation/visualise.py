# run train.py before visualise the results

import numpy as np
import os
import matplotlib.pyplot as plt

DATA_PATH = './data/datasets-promise12'
RESULT_PATH = './result' 

slices_to_plot = [2,5,8,11,14]  # must be smaller than total number of slices

# find all the available results
for filename in os.listdir(RESULT_PATH):
    if filename.endswith(".npy"):
        print('Label loaded: %s.' % os.path.join(RESULT_PATH, filename))
        label = np.load(os.path.join(RESULT_PATH, filename))[..., 0]
        image = np.load(os.path.join(DATA_PATH, "image_"+filename.split('_')[1]+".npy"))[::2, ::2, ::2]
        print(label.shape)
        plt.figure()
        plt.title(filename)
        for idx in range(len(slices_to_plot)):
            axs = plt.subplot(len(slices_to_plot),2,idx*2+1)
            axs.imshow(image[slices_to_plot[idx],:,:], cmap='gray')
            axs.axis('off')
            axs = plt.subplot(len(slices_to_plot),2,idx*2+2)
            axs.imshow(label[slices_to_plot[idx],:,:], cmap='gray')
            axs.axis('off')
        plt.ion()
        plt.show()
