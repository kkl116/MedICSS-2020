# Deep Learning for Medical Image Segmentation and Registration
A Project at [UCL Medical Image Computing Summer School (MedICSS)](https://medicss.cs.ucl.ac.uk/), 6 – 10 July 2020, London, UK.

## Tutors:  
Mon: Zac (zachary.baum.19@ucl.ac.uk)  
Tue: Yipeng (yipeng.hu@ucl.ac.uk)  
Wed: Mark (mark.pinnock.18@ucl.ac.uk)  
Thu: Qianye (qianye.yang.19@ucl.ac.uk)  

UCL Centre for Medical Image Computing  
Wellcome/EPSRC Centre for Interventional and Surgical Sciences  
University College London 2020

## System setup
### Install `Anaconda/Miniconda`. 
  See the official conda [instructions](https://docs.anaconda.com/anaconda/install/).  

### Create a conda virtual enviroment
```bash
conda create -n medicss tensorflow==2.0 matplotlib
```
  If you have a gpu available:
```bash
conda create -n medicss tensorflow-gpu==2.0 matplotlib
```

### Activate `medicss`
```bash
conda activate medicss
```
  See more details on [how to manage conda enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).  
  Learn more about [TensorFlow](https://www.tensorflow.org/)

## Medical image segmentation
`cd` to the `segmentation` folder - use it as the home directory.
### Download data
  Download data from a segmentation challenge [PROMISE12](https://promise12.grand-challenge.org/)
```bash
python data.py
```

### Train a segmentation network 
```bash
python train.py
```

### Plot the predicted segmentation on test data
```bash
python visualise.py
```

## Medical image registration
The open-source [DeepReg](https://github.com/ucl-candi/DeepReg/) is used.  
See the tutorials in the [DeepReg Documentation](https://ucl-candi.github.io/DeepReg/#/).




# The project 
## Introduction 
One of the most successful modern deep-learning applications in medical imaging is image segmentation. From neurological pathology in MR volumes to fetal anatomy in ultrasound videos, from cellular structures in microscopic images to multiple organs in whole-body

 CT scans, the list is ever expanding. This tutorial project will guide students to build and train a state-of-the-art convolutional neural network from scratch, then validate it on real patient data. The objective of this project is to obtain 1) basic understanding of machine learning approaches applied for medical image segmentation, 2) practical knowledge of essential components in building and testing deep learning algorithms, and 3) obtain hands-on experience in coding a deep segmentation network for real-world clinical applications.

Prerequisites: Python, GPU

## Potential topics
 1. Extension of the basic example, regularisation, data augmentation, new network architecture
 2. Comparison between a 2D network and a 3D network
 3. Find a way to utilise the unlabeled test data in training
 4. Join the challenge and be ranked 
