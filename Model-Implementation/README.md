
This file is dedicated for implementing the [Deep Fashion Analysis with Feature Map Upsampling and Landmark-driven Attention](https://drive.google.com/file/d/1Dyj0JIziIrTRWMWDfPOapksnJM5iPzEi/view)model in a non-GPU machine.

## Demo for how to use it

### 1. Import 

from model_loading import Label_Fashion_Image

### 2. Suppose all the image you want do classfication are in the following directory:

directory='/Users/zhang.xiaoya/Desktop/photos/'


### 3. Initiate an instance of the classification model:


p=Label_Fashion_Image()

### 4. Label all the image 

all_=p.all_img(directory)


all_.to_csv('result.csv')


### 5. In case you want do classification for single image,  there is no need to initiate another instance, just use the previous one:

file='q.jpg'

single_=p.single_img('q.jpg')
