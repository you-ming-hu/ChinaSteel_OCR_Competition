import pathlib
import matplotlib.pyplot as plt
import cv2

import Dataset

def overturn_image(model,save_path):
    data = Dataset.Test()
    save_path = pathlib.Path(save_path)
    for images,filepaths in data:
        predicts = model.predict(images)
        for filepath,predict in zip(filepaths,predicts):
            image = plt.imread(filepath)
            if predict:
                image = cv2.rotate(image, cv2.ROTATE_180)
            image = image[:,:,0]
            plt.imsave(save_path.joinpath(filepath.name).as_posix(),image,cmap='gray')