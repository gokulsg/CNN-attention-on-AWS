# Created Jun27th 2019 @Tokyo by xiaoya 
import pandas as pd
import numpy as np
import time 
import io
from PIL import Image
import requests
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from src.networks import WholeNetwork
import os
import torch

class Label_Fashion_Image:
    def __init__(self,path='../model/whole.pkl'):
        self.path=path
        m=torch.load(self.path,map_location='cpu')
        self.model = WholeNetwork()
        self.model.load_state_dict(m)
        #self.directory=directory
        self.attr_catagory={'Texture':1,'Fabric':2,'Shape':3,'Part':4,'Style':5}
        #self.attr_dataframe=get_attribute()
        #def get_attribute(self):
        with open('../benchmark1/Anno/list_attr_cloth.txt') as f:
            ret = []
            f.readline()
            f.readline()
            for line in f:
                line = line.split(' ')
                while line[-1].strip().isdigit() is False:
                    line = line[:-1]
                ret.append([
                    ' '.join(line[0:-1]).strip(),
                    int(line[-1])
                ])
        attr_type = pd.DataFrame(ret, columns=['attr_name', 'type'])
        attr_type['attr_index'] = ['attr_' + str(i) for i in range(1000)]
        attr_type.set_index('attr_index', inplace=True)
        self.attr_dataframe=attr_type
        
    def single_img(self,directory,filename):
        img = Image.open(directory + filename)
        # Now that we have an img, we need to preprocess it.
        # We need to:
        #       * resize the img, it is pretty big (~1200x1200px).
        #       * normalize it, as noted in the PyTorch pretrained models doc,
        #         with, mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        #       * convert it to a PyTorch Tensor.
        #
        # We can do all this preprocessing using a transform pipeline.
        #min_img_size = 224  
        # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
        transform_pipeline = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        img = transform_pipeline(img)
        img = img.unsqueeze(0)
        img = Variable(img)
        pred=self.model(img)
        pred_attr=pred['attr_output'].cpu().detach().numpy()
        pred_attr = np.split(pred_attr, pred_attr.shape[0])
        pred_attr = [x[0, 1, :] for x in pred_attr]
        pred_cata = pred['category_output']
        # get index for category
        category_type = pd.read_csv('../benchmark1/Anno/list_category_cloth.txt', skiprows=1, sep='\s+')
        category_name=list(category_type['category_name'])
        pred_cata=pred['category_output']
        cata_pred= pred_cata.sort(descending=True)[0].cpu().detach().numpy()[0][:5]
        cata_rank=pred_cata.sort(descending=True)[1].cpu().detach().numpy()[0][:5]
        cata_name=[category_name[i] for i in cata_rank]
        xy=dict()
        xy['category/attribute']=['category' for i in range(5)]
        xy['name']=cata_name
        xy['probability']=cata_pred
        cate_data=pd.DataFrame(xy)
        #print(cata_rank,cata_pred,cata_name)
        # get the index for top-5 attr
        pred_attr=pred['attr_output'].cpu().detach().numpy()
        pred_attr = np.split(pred_attr, pred_attr.shape[0])
        pred_attr = [x[0, 1, :] for x in pred_attr]
        attr=pd.DataFrame({'pred':pred_attr[0]})
        attr['attr_index'] = ['attr_' + str(i) for i in range(1000)]
        attr.set_index('attr_index', inplace=True)
        attr_pred=self.attr_dataframe.join(attr)
        attr_category=['Texture','Fabric','Shape','Part','Style']
        for i in range(5):
            i=i+1
            pred_3=attr_pred[attr_pred['type']==i]
            pred_3.reset_index(drop=True)
            xy=dict()
            xy['category/attribute']=[attr_category[i-1] for x in range(5)]
            xy['name']=list(pred_3.nlargest(5,['pred'])['attr_name'])
            xy['probability']=list(pred_3.nlargest(5,['pred'])['pred'])
            attr_data_part=pd.DataFrame(xy)
            cate_data=cate_data.append(attr_data_part,ignore_index=True)
        return cate_data
    
    def all_img(self,directory):
        '''
        this target directory can only contain imagr data
        ''' 
        file_lis=[]
        all_data=pd.DataFrame()
        for filename in os.listdir(directory):
            file_lis.append(filename)
            single_data=self.single_img(directory,filename)
            single_data['filename']=[filename for i in range(single_data.shape[0])]
            all_data=all_data.append(single_data,ignore_index=True)
        return all_data
