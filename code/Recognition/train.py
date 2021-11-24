import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from sklearn import model_selection
import io
import math

import FLAGS
import INVARIANT
import Dataset
import utils

FLAGS.CHECK()

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

log_path = pathlib.Path(FLAGS.LOGGING.PATH).joinpath('model_'+str(FLAGS.LOGGING.MODEL_NAME),'trial_'+str(FLAGS.LOGGING.TRIAL_NUMBER))
log_path.mkdir(parents=True)
steps_per_log = FLAGS.LOGGING.SAMPLES_PER_LOG//FLAGS.DATA.TRAIN.TRAIN_BATCH_SIZE

total_epochs = FLAGS.EPOCHS.TOTAL
warmup_epochs = FLAGS.EPOCHS.WARMUP
steps_per_epoch = len(train_data)
warmup_steps = steps_per_epoch * warmup_epochs

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps, gamma=-0.5):
        super().__init__()
        self.max_lr = tf.cast(max_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps,tf.float32)
        assert gamma<0
        self.gamma = tf.cast(gamma,tf.float32)

    def __call__(self, step):
        arg1 = step ** self.gamma
        arg2 = step * (self.warmup_steps ** (self.gamma-1))
        return self.max_lr * (self.warmup_steps**-self.gamma) * tf.math.minimum(arg1, arg2)

schedule = CustomSchedule(max_lr,warmup_steps,schedule_gamma)
optimizer = optimizer_type(schedule)

label_smoothing = FLAGS.LOSS.LABEL_SMOOTHING

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

val_metrics = []

CE_metric_name = 'Cross Entropy loss'
train_CE_metric = tf.keras.metrics.Mean(name=CE_metric_name)
val_CE_metric = tf.keras.metrics.Mean(name=CE_metric_name)
best_val_CE_metric_value = tf.Variable(0.)
val_metrics.append((val_CE_metric,best_val_CE_metric_value))

correct_ratio_metric_name = 'Correct ratio'
train_correct_ratio_metric = tf.keras.metrics.Mean(name=correct_ratio_metric_name)
val_correct_ratio_metric = tf.keras.metrics.Mean(name=correct_ratio_metric_name)
best_val_correct_ratio_metric_value = tf.Variable(0.)
val_metrics.append((val_correct_ratio_metric,best_val_correct_ratio_metric_value))

all_correct_metric_name = 'All correct'
train_all_correct_metric = tf.keras.metrics.Mean(name=all_correct_metric_name)
val_all_correct_metric = tf.keras.metrics.Mean(name=all_correct_metric_name)
best_val_all_correct_metric_value = tf.Variable(0.)
val_metrics.append((val_all_correct_metric,best_val_all_correct_metric_value))

train_writer = tf.summary.create_file_writer(log_path.joinpath('summary','train').as_posix())
validation_writer = tf.summary.create_file_writer(log_path.joinpath('summary','validation').as_posix())

with train_writer.as_default(0):
    config = FLAGS.GET_CONFIG()
    note = f'Note:  \n{FLAGS.LOGGING.NOTE}'
    tf.summary.text('Detail',[config,note])

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
        train_CE_metric.update_state(loss)
        train_correct_ratio_metric.update_state(correct_ratio)
        train_all_correct_metric.update_state(all_correct)
        step += 1
        if step%steps_per_log == 0:
            with train_writer.as_default(step):
                tf.summary.scalar(train_CE_metric.name, train_CE_metric.result())
                tf.summary.scalar(train_correct_ratio_metric.name, train_correct_ratio_metric.result())
                tf.summary.scalar(train_all_correct_metric.name, train_all_correct_metric.result())
                tf.summary.scalar('Learning rate', schedule(step))
            train_CE_metric.reset_state()
            train_correct_ratio_metric.reset_state()
            train_all_correct_metric.reset_state()
    
    for batch_data in val_data:
        image,sequence = batch_data
        raw_predict = model(image)
        loss = CE_loss(sequence,raw_predict,label_smoothing,corpus_size)
        correct_ratio, all_correct = Accuracy(sequence,raw_predict)
        val_CE_metric.update_state(loss)
        val_correct_ratio_metric.update_state(correct_ratio)
        val_all_correct_metric.update_state(all_correct)
    
    save_weights_path = log_path.joinpath('weights',f'{e:0>4}')
    save_weights_path.mkdir(parents=True)
    model.save_weights(save_weights_path.joinpath('weights').as_posix())
    
    utils.log_best_val_metrics(e,step,val_metrics,save_weights_path,train_writer,validation_writer)
    
    ncols = FLAGS.LOGGING.TEST_IMAGE_COLUMNS
    nrows = math.ceil(len(test_data)/ncols)
    test_img_index = 1
    
    fig = plt.figure(figsize=(25,25),dpi=10)
    plt.title('Predict string\n',fontsize=40)
    plt.axis(False)
    for test_images, test_filepaths in test_data:
        test_preds = model.predict(test_images)
        for test_filepath,test_pred in zip(test_filepaths,test_preds):
            test_img = plt.imread(test_filepath.as_posix())/255
            fig.add_subplot(nrows,ncols,test_img_index)
            plt.imshow(test_img,cmap='gray')
            plt.title(test_pred,fontsize=20)
            plt.axis(False)
            test_img_index += 1
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=60)
    plt.close()
    buf.seek(0)
    test_img = tf.image.decode_jpeg(buf.getvalue(), channels=3)
    test_img = test_img[None,...]
    with train_writer.as_default(step):
        tf.summary.image('test data prediction',test_img)