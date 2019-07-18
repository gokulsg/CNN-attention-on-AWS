# Setup AWS for training deep learning model using pytorch
This repo will adress the following aspects of using AWS for training deep learning models 
* **General setup for AWS EC2 (Elastic Compute Cloud) and S3 Storage Classes for Pytorch**
  * **The [code](https://github.com/fdjingyuan/Deep-Fashion-Analysis-ECCV2018) and [dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) for paper [Deep Fashion Analysis with Feature Map Upsampling and Landmark-driven Attention (2018)](https://drive.google.com/file/d/1Dyj0JIziIrTRWMWDfPOapksnJM5iPzEi/view) is taken as example to show the workflow** 
  * Manage the remote terminal sessions with tmux.
* Data preprocessing for duplicated attributes.
* Load the GPU-trained model and run on local machine.


## Data preprocessing for duplicated attributes.

After following the [post](https://github.com/fdjingyuan/Deep-Fashion-Analysis-ECCV2018/)'s creating info.csv step, 
it is necessary to do some data cleaning since the original DeepFashion Dataset has a lot duplicated labels, 
such as : stripe, stripes, striped are listed as separate attributes.

The demostration for running the scripts are inside the data_cleaning file.


## Load GPU-trained model on local machine
The workflow for this specific example will provide a standard practice for loading trained model and put it into use on your local machine. the code is in file Model-Implementation.

The demostration for using the classification model is in the readme.md inside Model-Implementation file.

