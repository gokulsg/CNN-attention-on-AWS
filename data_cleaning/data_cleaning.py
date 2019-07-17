#created by xiaoya @ tokyo July 12th 2019
# edit by xiaoya on July 16th 
import pandas as pd


class merge_onehotvector_columns:
    '''
    this code is designed for merging one hot vector columns when some items are duplicated
    '''
    def __init__(self,attr_min_num,info_file,merge_column_file):
        self.attr_min_num=attr_min_num
        self.file=info_file
        self.merge_column_file= merge_column_file
        # image selection
        self.df=pd.read_csv(self.file)
        self.df=self.df.loc[self.df[['attr_{}'.format(n) for n in range(1000)]].sum(axis=1)>self.attr_min_num]
        # not yet re-index 
        # answer: less than 2 attri are 48021 
        # answer : more than 2 attri are between 90000-100000
        self.df.reset_index(inplace=True)
        
    def merge_dic(self):
        with open(self.merge_column_file) as f:
            lines = f.readlines()
            dic_items=[i.strip().split(",") for i in lines]
            dic_keys=[i[0] for i in dic_items]
        return dict(zip(dic_keys,dic_items))

    def merge_columns(self): 
        count=0
        for keys,values in self.merge_dic().items():
            count+=1 
            # get the sum
            self.df['attr_{}'.format(keys)]=self.df[['attr_{}'.format(i) for i in values]].sum(axis=1)
            # if the value bigger than 0, set it to 1
            self.df['attr_{}'.format(keys)]=[int(i) for i in self.df['attr_{}'.format(keys)]>0]
            #set all the other duplicated column to 0
            for i in values:
                self.df['attr_{}'.format(i)]=0
            if count%10==0:
                print (count)
        self.df.to_csv('info_processed.csv'.format(self.attr_min_num),index=False)
        print('finished merging for',self.df.shape[0],'images')
