import pathlib
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

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

def log_best_val_metrics(epoch,step,metrics,save_weights_path,train_writer,validation_writer):
    for metric,best_value in metrics:
        current_loss = metric.result()
        with validation_writer.as_default(step):
            tf.summary.scalar(metric.name, current_loss)
        if epoch == 0:
            best_value.assign(current_loss)
        else:
            if current_loss < best_value:
                best_value.assign(current_loss)
                with train_writer.as_default(step):
                    context = '  \n'.join([f'Epoch: {epoch}',f'Step: {step}', f'Loss: {best_value}', f"Weight: {save_weights_path.joinpath('weights').as_posix()}"])
                    tf.summary.text(metric.name,context)
        metric.reset_state()