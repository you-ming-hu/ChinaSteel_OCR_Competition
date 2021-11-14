import numpy as np
import FLAGS
import Dataset
import pandas as pd

corpus = FLAGS.CORPUS
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
