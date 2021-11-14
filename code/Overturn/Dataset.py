import pathlib
import sklearn
import numpy as np
import tensorflow as tf
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
        file = tf.io.read_file(path.as_posix())
        image = tf.io.decode_jpeg(file)
        image = image[:,:,0]
        image = image[:,:,None]
        image = tf.cast(image,tf.float64)
        return image

    def to_output_image(self,image):
        h,w = INVARIANT.DATASET_IMAGE_SHAPE
        image = tf.image.resize_with_pad(image,h,w)
        image = image/255
        return image
    
    def random_shape(self,image):
        new_shape = 1.2**np.random.choice(np.linspace(-1,1,51),2) * image.shape[:2]
        image = tf.image.resize(image,new_shape)
        return image

    def random_brightness(self,image):
        b = np.random.randint(-50,50)
        image = image + b
        image = tf.clip_by_value(image,0,255)
        return image

    def random_contrast(self,image):
        g = np.random.choice(np.linspace(-0.5,0.5,100))
        m = 255/2
        diff = image-m
        image = (m + diff*tf.math.abs(m/(diff+1e-10))**g)
        return image

    def image_arg(self,image):
        if bool(np.random.binomial(1,0.8)):
            image = self.random_shape(image)
        if bool(np.random.binomial(1,0.8)):
            image = self.random_contrast(image)
        if bool(np.random.binomial(1,0.8)):
            image = self.random_brightness(image)
        return image

class Train(Base):
    def __init__(self,table,batch_size,is_validation):
        super().__init__('',batch_size)
        self.table = table
        self.is_validation = is_validation
    
    def __len__(self):
        return self.table.index.size
    
    def __iter__(self):
        self.count = 0
        if not self.is_validation:
            self.table = sklearn.utils.shuffle(self.table)
        return self
    
    def __next__(self):
        if self.count < self.table.index.size:
            batch_image = []
            batch_overturn_label = []
            for _,row in self.table.iloc[self.count:self.count+self.batch_size].iterrows():
                image = self.get_image(row['filepath'],not self.is_validation)
                overturn_label = row['overturn']
                if overturn_label == 0:
                    if bool(np.random.binomial(1,0.5)):
                        image = tf.image.rot90(image,2)
                        overturn_label = 1
                else:
                    if bool(np.random.binomial(1,0.5)):
                        image = tf.image.rot90(image,2)
                        overturn_label = 0
                batch_image.append(image)
                batch_overturn_label.append(overturn_label)
            batch_image = tf.stack(batch_image,axis=0)
            batch_overturn_label = tf.stack(batch_overturn_label,axis=0)
            self.count += self.batch_size
            return batch_image,batch_overturn_label
        else:
            raise StopIteration    
        
class Test(Base):
    def __init__(self,image_path,batch_size):
        super().__init__(image_path,batch_size)
        self.image_paths = list(self.image_path.iterdir())
        
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
            batch_image = tf.stack(batch_image,axis=0)
            self.count += self.batch_size
            return batch_image,batch_filepath
        else:
            raise StopIteration