import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from sklearn import model_selection
import io

import FLAGS
import Dataset
import utils

FLAGS.CHECK()

data_path = pathlib.Path(FLAGS.DATA.TRAIN.DATA_PATH)
normal = pd.DataFrame(list(data_path.joinpath('normal').iterdir()),columns=['filepath'])
normal['overturn'] = 0
overturn = pd.DataFrame(list(data_path.joinpath('overturn').iterdir()),columns=['filepath'])
overturn['overturn'] = 1

val_ratio = FLAGS.DATA.TRAIN.VALIDATION_SPLIT_RATIO
split_random_state = FLAGS.DATA.TRAIN.VALIDATION_SPLIT_RANDOM_STATE
train_normal, val_normal = model_selection.train_test_split(normal, test_size=val_ratio, random_state=split_random_state)
train_overturn, val_overturn = model_selection.train_test_split(overturn, test_size=val_ratio, random_state=split_random_state)

train_table = train_normal.append(train_overturn,ignore_index=True)
val_table = val_normal.append(val_overturn,ignore_index=True)

train_data = Dataset.Train(train_table)
validation_data = Dataset.Train(val_table)
test_data = iter(Dataset.Test())

train_data = Dataset.Train(
    table = train_table,
    batch_size = FLAGS.DATA.TRAIN.TRAIN_BATCH_SIZE,
    is_validation = False)
    
val_data = Dataset.Train(
    table = val_table,
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
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=label_smoothing)

val_metrics = []

bce_metric_name = 'BCE loss'
train_bce_metric = tf.keras.metrics.BinaryCrossentropy(name=bce_metric_name ,from_logits=False, label_smoothing=label_smoothing)
val_bce_metric = tf.keras.metrics.BinaryCrossentropy(name=bce_metric_name, from_logits=False, label_smoothing=label_smoothing)
best_val_bce_metric_value = tf.Variable(0)
val_metrics.append((val_bce_metric,best_val_bce_metric_value))

accuracy_metric_name = 'Accuracy'
train_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name=accuracy_metric_name, threshold=0.5)
val_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name=accuracy_metric_name, threshold=0.5)
best_val_accuracy_metric_value = tf.Variable(0)
val_metrics.append((val_accuracy_metric,best_val_accuracy_metric_value))

train_writer = tf.summary.create_file_writer(log_path.joinpath('summary','train').as_posix())
validation_writer = tf.summary.create_file_writer(log_path.joinpath('summary','validation').as_posix())

with train_writer.as_default(0):
    config = FLAGS.GET_CONFIG()
    note = f'Note:  \n{FLAGS.LOGGING.NOTE}'
    tf.summary.text('Detail',[config,note])

step = 1
for e in range(total_epochs):
    for batch_data in train_data:
        image,overturn_label = batch_data
        with tf.GradientTape() as tape:
            predict = model(image)
            loss = bce_loss(overturn_label,predict)   
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_bce_metric.update_state(overturn_label,predict)
        train_accuracy_metric.update_state(overturn_label,predict)
        
        step += 1
        if step%steps_per_log == 0:
            with train_writer.as_default(step):
                tf.summary.scalar(train_bce_metric.name, train_bce_metric.result())
                tf.summary.scalar(train_accuracy_metric.name, train_accuracy_metric.result())
                tf.summary.scalar('Learning rate', schedule(step))
            train_bce_metric.reset_state()
            train_accuracy_metric.reset_state()
    
    for batch_data in validation_data:
        image,overturn_label = batch_data
        predict = model(image)
        loss = bce_loss(overturn_label,predict)
        val_bce_metric.update_state(overturn_label,predict)
        val_accuracy_metric.update_state(overturn_label,predict)

    save_weights_path = log_path.joinpath('weights',f'{e:0>4}')
    save_weights_path.mkdir()
    model.save_weights(save_weights_path.joinpath('weights').as_posix())

    utils.log_best_val_metrics(e,step,val_metrics,save_weights_path,train_writer,validation_writer)
    
    test_records = []
    test_imgs,test_filepaths = next(test_data)
    test_preds = model.predict(test_imgs)
    for test_filepath,test_pred in zip(test_filepaths,test_preds):
        test_img = plt.imread(test_filepath.as_posix())
        plt.imshow(test_img,cmap='gray')
        plt.axis(False)
        plt.title('Is rotated 180 degree?\nPrediction: ' + str(test_pred),fontsize=10)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        test_img = tf.image.decode_png(buf.getvalue(), channels=3)
        test_records.append(test_img)
    test_records = tf.stack(test_records,axis=0)
    with train_writer.as_default(step):
        tf.summary.image('test data prediction',test_records)