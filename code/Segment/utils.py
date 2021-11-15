import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pathlib

import INVARIANT
import Dataset

def get_rectangle_points(img,bboxes):
    bboxes[[0,2]] = bboxes[[0,2]] * img.shape[1]
    bboxes[[1,3]] = bboxes[[1,3]] * img.shape[0]
    bboxes[[0,1]] = bboxes[[0,1]] - bboxes[[2,3]]/2
    bboxes[[2,3]] = bboxes[[2,3]] + bboxes[[0,1]]
    bboxes = np.reshape(bboxes[[0,1,2,1,2,3,0,3]],[4,2])
    bboxes = bboxes.astype(int)
    return bboxes

def draw_bbox_on_image(img,bboxes,color = (1, 1, 1),lw=1):
    points = get_rectangle_points(img,bboxes)
    img = cv2.polylines(img, [points], True, color, lw)
    return img
    
def segment(model,input_path,output_path,batch_size,tolerance,save_compare=False):
    data = Dataset.Test(input_path,batch_size)
    output_path = pathlib.Path(output_path)
    record = pd.DataFrame(columns=['filename','xmin','xmax','ymin','ymax','xcentre','ycentre','w','h','aspect'])
    for imgs,filepaths in data:
        preds = model.predict(imgs)
        preds = preds.numpy()
        preds[:,[2,3]] = preds[:,[2,3]]*tolerance
        h,w = imgs.shape[1:3]
        for filepath,img,pred in zip(filepaths,imgs,preds):
            xmin = np.clip(int((pred[0] - pred[2]/2)*w),0,w-1)
            xmax = np.clip(int((pred[0] + pred[2]/2)*w),0,w-1)
            ymin = np.clip(int((pred[1] - pred[3]/2)*h),0,h-1)
            ymax = np.clip(int((pred[1] + pred[3]/2)*h),0,h-1)
            record = record.append({
                'filename':filepath.name,
                'xmin':xmin,
                'xmax':xmax,
                'ymin':ymin,
                'ymax':ymax,
                'xcentre':(xmin+xmax)/2,
                'ycentre':(ymin+ymax)/2,
                'w':xmax-xmin,
                'h':ymax-ymin,
                'aspect':(xmax-xmin)/(ymax-ymin)
                },ignore_index=True)

            segment_img = np.squeeze(img[ymin:ymax,xmin:xmax])
            plt.imsave(output_path.joinpath('image',filepath.name).as_posix(),segment_img,cmap='gray')

            if save_compare:
                bbox = np.array([[[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]])
                img = cv2.polylines(img, bbox, True, (1,1,1), 5)
                img = np.squeeze(img)
                plt.imsave(output_path.joinpath('compare',filepath.name).as_posix(),img,cmap='gray')

    record.to_csv(output_path.joinpath('record.csv').as_posix(),index=False,encoding='utf8')
    return record

def preprocess_table(table):
    def coor_polygon_to_rect(polygon_coor):
        image_shape = INVARIANT.RAW_IMAGE_SHAPE
        polygon_coor = polygon_coor.values.reshape((4,2))
        polygon_coor = np.round(polygon_coor.astype(float)).astype(int)
        xywh = np.array(cv2.boundingRect(polygon_coor))
        xywh = xywh.astype(float)
        xywh[:2] = xywh[:2] + xywh[2:]/2
        xywh[[0,2]] = xywh[[0,2]]/image_shape[1]
        xywh[[1,3]] = xywh[[1,3]]/image_shape[0]
        return pd.Series(xywh,index=['x','y','w','h'])
    bboxes = table[['top right x', 'top right y','bottom right x','bottom right y','bottom left x', 'bottom left y','top left x','top left y']]
    xywh = bboxes.apply(coor_polygon_to_rect,axis=1)
    table = pd.concat([table,xywh],axis=1)
    return table
