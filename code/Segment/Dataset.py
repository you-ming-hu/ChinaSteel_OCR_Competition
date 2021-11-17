import pathlib
import sklearn
import numpy as np
import cv2
import matplotlib.pyplot as plt

import INVARIANT

class Base:
    def __init__(self,image_path,batch_size):
        self.image_path = pathlib.Path(image_path)
        self.batch_size = batch_size

    def get_image(self,path,arg):
        image = self.read_image(path)
        if arg:
            image = self.image_arg(image)
        image = self.to_output_image(image)
        return image

    def read_image(self,path):
        image = plt.imread(path)
        image = cv2.resize(image,INVARIANT.DATASET_IMAGE_SHAPE,interpolation=cv2.INTER_LINEAR)
        image = image[:,:,None]
        image = image.astype(int)
        return image
    
    def to_output_image(self,image):
        image = image/255
        return image

    def random_brightness(self,image):
        b = np.random.randint(-25,25)
        image = image + b
        image = np.clip(image,0,255)
        return image
        
    def random_contrast(self,image):
        g = np.random.choice(np.linspace(-0.75,0.25,75))
        m = 255/2
        diff = image-m
        return (m + diff*abs(m/(diff))**g).astype(int)

    def image_arg(self,image):
        if bool(np.random.binomial(1,0.5)):
            image = self.random_contrast(image)
        if bool(np.random.binomial(1,0.5)):
            image = self.random_brightness(image)
        return image    


class Train(Base):
    def __init__(self,table,image_path,batch_size,is_validation):
        super().__init__(image_path,batch_size)
        self.table = table
        self.is_validation = is_validation
        
    def __iter__(self):
        self.count = 0
        if not self.is_validation:
            self.table = sklearn.utils.shuffle(self.table)
        return self

    def __next__(self):
        if self.count < self.table.index.size:
            batch_image = []
            batch_xywh = []
            for _,row in self.table.iloc[self.count:self.count+self.batch_size].iterrows():
                image = self.get_image(self.image_path.joinpath(row['filename']).with_suffix('.jpg'),not self.is_validation)
                xywh = row[['x','y','w','h']].values.astype(float)
                batch_image.append(image)
                batch_xywh.append(xywh)
            batch_image = np.stack(batch_image,axis=0)
            batch_xywh = np.stack(batch_xywh,axis=0)
            self.count += self.batch_size
            return batch_image,batch_xywh
        else:
            raise StopIteration

    def __len__(self):
        return self.table.index.size // self.batch_size + 1
    
    def __getitem__(self,key):
        return __class__(self.table.iloc[key],self.image_path,self.batch_size,self.is_validation)

class Test(Base):
    def __init__(self,image_path,batch_size):
        super().__init__(image_path,batch_size)
        self.image_paths = list(self.image_path.iterdir())
        
    def __len__(self):
        return len(self.image_paths)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_paths):
            batch_image = []
            batch_filepath = []
            for path in self.image_paths[self.count:self.count+self.batch_size]:
                image = self.get_image(path,False)
                batch_image.append(image)
                batch_filepath.append(path)
            batch_image = np.stack(batch_image,axis=0)
            self.count += self.batch_size
            return batch_image,batch_filepath
        else:
            raise StopIteration