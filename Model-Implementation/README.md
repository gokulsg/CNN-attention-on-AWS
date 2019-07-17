# demo for how to use it

## Import 

from model_loading import Label_Fashion_Image

## Suppose all the image you want do classfication are in the following directory:

directory='/Users/zhang.xiaoya/Desktop/photos/'


## Initiate an instance of the classification model:


p=Label_Fashion_Image()

## Label all the image 

all_=p.all_img(directory)


all_.to_csv('result.csv')


## In case you want do classification for single image,  there is no need to initiate another instance, just use the previous one:

file='q.jpg'

single_=p.single_img('q.jpg')
