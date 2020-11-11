This repo will adress the following aspects of using AWS for training deep learning models 

* **General setup for AWS EC2 (Elastic Compute Cloud) and S3 Storage Classes for training deep learning model using Pytorch**
  * **The implementation for paper [Deep Fashion Analysis with Feature Map Upsampling and Landmark-driven Attention (2018)](https://drive.google.com/file/d/1Dyj0JIziIrTRWMWDfPOapksnJM5iPzEi/view) is taken as example to show the workflow** 
  * Manage the remote terminal sessions with tmux.
* Data preprocessing for duplicated attributes.
* Load the GPU-trained model and run on local machine.


## General setup for AWS EC2 (Elastic Compute Cloud) and S3 Storage Bucket.


## Data preprocessing for duplicated attributes.

After following the [post](https://github.com/fdjingyuan/Deep-Fashion-Analysis-ECCV2018/)'s creating info.csv step, 
it is necessary to do some data cleaning since the original DeepFashion Dataset has a lot duplicated labels, 
such as : stripe, stripes, striped are listed as separate attributes.

The demostration for running the scripts are inside the data_cleaning file.

Using preprocessed data for training could significantly improve the accuracy, the results are follows:



The following table shows the category classification and attribute prediction results on the DeepFashion dataset for the original dataset and our pre-processed dataset. The two numbers in each cell stands for top-3 and top-5 accuracy. 

| Methods         | Category               | Texture                | Fabric         | Shape                  | Part                    | Style              | All                |
|:---------------:|:----------------------:|:----------------------:|:--------------:|:----------------------:|:-----------------------:|:------------------:|:------------------:|
| [Liu.et al.](https://drive.google.com/file/d/1Dyj0JIziIrTRWMWDfPOapksnJM5iPzEi/view)       | 91.16 \| 96.12 | 56.17 \| 65.83 | 43.20 \| 53.52 | 58.28 \| 67.80 | 46.97 \| 57.42  | 68.82 \| 74.13 | 54.69 \| 63.74 |
| Ours       | 90.99 \| 95.88 | 69.92 \| 78.86 | 50.42 \| 60.91 | 64.02 \| 72.43 | 59.03 \| 68.87  | 37.42 \| 46.40 | 31.13 \| 46.40 |





## Load GPU-trained model on local machine
The workflow for this specific example will provide a standard practice for loading trained model and put it into use on your local machine. the code is in file Model-Implementation.

The demostration for using the classification model is in the readme.md inside Model-Implementation file.

