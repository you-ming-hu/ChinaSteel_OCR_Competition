import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from sklearn import model_selection

import FLAGS
import INVARIANT
import Dataset

corpus_size = len(INVARIANT.CORPUS)

table = pd.read_csv(FLAGS.DATA.TRAIN.TABLE_PATH)

train_table, val_table = model_selection.train_test_split(
    table,
    test_size=FLAGS.DATA.TRAIN.VALIDATION_SPLIT_RATIO ,
    random_state=FLAGS.DATA.TRAIN.VALIDATION_SPLIT_RANDOM_STATE)

train_data = Dataset.Train(
    table = train_table,
    image_path = FLAGS.DATA.TRAIN.IMAGE_PATH,
    batch_size = FLAGS.DATA.TRAIN.TRAIN_BATCH_SIZE,
    is_validation = False)
    
val_data = Dataset.Train(
    table = val_table,
    image_path = FLAGS.DATA.TRAIN.IMAGE_PATH,
    batch_size = FLAGS.DATA.TRAIN.VAL_BATCH_SIZE,
    is_validation = True)

test_data = Dataset.Test(
    image_path = FLAGS.DATA.TEST.IMAGE_PATH,
    batch_size = FLAGS.DATA.TEST.BATCH_SIZE)
test_data = iter(test_data)

model = FLAGS.MODEL

optimizer_type = FLAGS.OPTIMIZER.TYPE
max_lr = FLAGS.OPTIMIZER.MAX_LEARNING_RATE
schedule_gamma = FLAGS.OPTIMIZER.SCHEDULE_GAMMA

log_path = pathlib.Path(FLAGS.LOGGING.PATH).joinpath(FLAGS.LOGGING.MODEL_NAME,str(FLAGS.LOGGING.TRIAL_NUMBER))
log_path.mkdir(parents=True)
steps_per_log = FLAGS.LOGGING.STEPS_PER_LOG

total_epochs = FLAGS.EPOCHS.TOTAL
warmup_epochs = FLAGS.EPOCHS.WARMUP
steps_per_epoch = len(train_data)
warmup_steps = steps_per_epoch * warmup_epochs



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps, gamma=-0.5):
        super().__init__()
        self.max_lr = tf.cast(max_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps,tf.float32)
        self.gamma = tf.cast(gamma,tf.float32)

    def __call__(self, step):
        arg1 = step ** self.gamma
        arg2 = step * (self.warmup_steps ** (self.gamma-1))
        return self.max_lr * (self.warmup_steps**-self.gamma) * tf.math.minimum(arg1, arg2)

# model = FLAGS.MODEL

# record_weights_path = pathlib.Path(FLAGS.TRAIN.RECORD_WEIGHTS_PATH)

# max_lr = FLAGS.TRAIN.OPTIMIZER.MAX_LR
# optimizer_type = FLAGS.TRAIN.OPTIMIZER.TYPE

total_epochs = FLAGS.TRAIN.EPOCH.TOTAL
warmup_epochs = FLAGS.TRAIN.EPOCH.WARMUP
steps_per_epoch = len(train_data)
warmup_steps = steps_per_epoch * warmup_epochs

label_smoothing = FLAGS.TRAIN.LABEL_SMOOTHING

schedule = CustomSchedule(max_lr,warmup_steps,FLAGS.TRAIN.OPTIMIZER.SCHEDULE_GAMMA)
optimizer = optimizer_type(schedule)

def CE_loss(label,raw_predict,smoothing,corpus_size):
    one_hot_label = tf.one_hot(label,axis=-1,depth=corpus_size)
    smooth_one_hot_label = one_hot_label * (1-smoothing)
    smooth_one_hot_label = smooth_one_hot_label + smoothing/corpus_size
    smooth_one_hot_label = tf.stop_gradient(smooth_one_hot_label)
    CE = tf.nn.softmax_cross_entropy_with_logits(labels=smooth_one_hot_label, logits=raw_predict, axis=-1)
    CE = tf.math.reduce_sum(CE,axis=-1)
    return CE

def Accuracy(label,raw_predict):
    predict = tf.argmax(raw_predict,axis=-1)
    predict = tf.cast(predict,tf.int32)
    match = (label == predict)
    correct_ratio = tf.math.reduce_mean(tf.cast(match,tf.float64),axis=-1)
    all_correct = tf.math.reduce_all(match,axis=-1)
    all_correct = tf.cast(all_correct,tf.float64)
    return correct_ratio, all_correct

cross_entropy_metric = tf.keras.metrics.Mean()
correct_ratio_metric = tf.keras.metrics.Mean()
all_correct_metric = tf.keras.metrics.Mean()

step = 1
for e in range(total_epochs):
    for batch_data in train_data:
        image,sequence = batch_data
        with tf.GradientTape() as tape:
            raw_predict = model(image)
            loss = CE_loss(sequence,raw_predict,label_smoothing,corpus_size)
            mean_loss = tf.math.reduce_mean(loss)
        gradients = tape.gradient(mean_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        correct_ratio, all_correct = Accuracy(sequence,raw_predict)
        
        cross_entropy_metric.update_state(loss)
        correct_ratio_metric.update_state(correct_ratio)
        all_correct_metric.update_state(all_correct)
        
        step += 1
        if step%100 == 0:
            print(f'STEP: {step:>6} TRAIN loss CE: {cross_entropy_metric.result():.5f}, Correct ratio {correct_ratio_metric.result():.5f}, All correct: {all_correct_metric.result():.5f}')
        
    cross_entropy_metric.reset_state()
    correct_ratio_metric.reset_state()
    all_correct_metric.reset_state()
    
    print('='*20,f'EPOCH: {e}','='*20)
    for batch_data in validation_data:
        image,sequence = batch_data
        raw_predict = model(image)
        loss = CE_loss(sequence,raw_predict,label_smoothing,corpus_size)
        correct_ratio, all_correct = Accuracy(sequence,raw_predict)
        cross_entropy_metric.update_state(loss)
        correct_ratio_metric.update_state(correct_ratio)
        all_correct_metric.update_state(all_correct)
    print(f'VALIDATION loss CE: {cross_entropy_metric.result():.5f}, Correct ratio {correct_ratio_metric.result():.5f}, All correct: {all_correct_metric.result():.5f}')
    current_validation_loss = cross_entropy_metric.result()
    if e == 0:
        min_validation_loss = current_validation_loss
        suffix = ''
    else:
        if  current_validation_loss < min_validation_loss:
            min_validation_loss = current_validation_loss
            suffix = '_SOTA'
        else:
            suffix = ''
    
    save_weights_path = record_weights_path.joinpath(f'{e:0>4}_{current_validation_loss:.5f}{suffix}'.replace('.','p'))
    save_weights_path.mkdir()
    model.save_weights(save_weights_path.joinpath('weights').as_posix())
    
    cross_entropy_metric.reset_state()
    correct_ratio_metric.reset_state()
    all_correct_metric.reset_state()

    test_imgs = next(test_data)
    pred = model.predict(test_imgs)
    for img,p in zip(test_imgs,pred):
        print(p)
        plt.imshow(img,cmap='gray')
        plt.show()