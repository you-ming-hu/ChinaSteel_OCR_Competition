import numpy as np
import pandas as pd
import tensorflow as tf

import INVARIANT
import Dataset

corpus = INVARIANT.CORPUS
char2token_dict = {char:token for token,char in enumerate(corpus)}
token2char_dict = {token:char for token,char in enumerate(corpus)}

def char2token(seq):
    return np.array(list(map(lambda x: char2token_dict[x],seq)))

def token2char(seq):
    return ''.join(map(lambda x: token2char_dict[x],seq)).replace(' ','')

def recognize(model,image_path,batch_size=8):
    data = Dataset.Recognition(image_path, batch_size)
    table = pd.DataFrame(columns=['id','text'])
    for images, filenames in data:
        predicts = model.predict(images)
        table = table.append(pd.DataFrame({'id':filenames,'text':predicts}),ignore_index=True)
    return table


def log_best_val_metrics(epoch,step,metrics,save_weights_path,train_writer,validation_writer):
    for metric,best_value in metrics:
        current_loss = metric.result()
        with validation_writer.as_default(step):
            tf.summary.scalar(metric.name, current_loss)
        if epoch == 0:
            best_value.assign(current_loss)
        else:
            if metric.name in ['Cross Entropy loss']:
                if current_loss < best_value:
                    best_value.assign(current_loss)
                    with train_writer.as_default(step):
                        context = '  \n'.join([f'Epoch: {epoch}',f'Step: {step}', f'{metric.name}: {best_value.numpy()}', f"Weight: {save_weights_path.joinpath('weights').as_posix()}"])
                        tf.summary.text('best '+metric.name,context)
            else:
                if current_loss > best_value:
                    best_value.assign(current_loss)
                    with train_writer.as_default(step):
                        context = '  \n'.join([f'Epoch: {epoch}',f'Step: {step}', f'{metric.name}: {best_value.numpy()}', f"Weight: {save_weights_path.joinpath('weights').as_posix()}"])
                        tf.summary.text('best '+metric.name,context)
        metric.reset_state()