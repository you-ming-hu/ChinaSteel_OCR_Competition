import tensorflow as tf
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
import pathlib

import FLAGS
import utils
import Dataset

FLAGS.CHECK()

table = pd.read_csv(FLAGS.DATA.TRAIN.TABLE_PATH)
table = utils.preprocess_table(table)

if FLAGS.DATA.TRAIN.DROP_BAD_BBOX_DATA:
    table = table.loc[table['w']/table['h']>3]

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
        assert gamma<0
        self.gamma = tf.cast(gamma,tf.float32)

    def __call__(self, step):
        arg1 = step ** self.gamma
        arg2 = step * (self.warmup_steps ** (self.gamma-1))
        return self.max_lr * (self.warmup_steps**-self.gamma) * tf.math.minimum(arg1, arg2)

schedule = CustomSchedule(max_lr,warmup_steps,schedule_gamma)
optimizer = optimizer_type(schedule)

def GIOU_loss(bboxes1,bboxes2):
    bboxes1 = tf.cast(bboxes1,tf.float32)
    bboxes2 = tf.cast(bboxes2,tf.float32)
    
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat([bboxes1[..., :2] - bboxes1[..., 2:] * 0.5, bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,],axis=-1)
    bboxes2_coor = tf.concat([bboxes2[..., :2] - bboxes2[..., 2:] * 0.5, bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,],axis=-1)
    
    
    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)
    return 1 - giou

train_metric = tf.keras.metrics.Mean()
val_metric = tf.keras.metrics.Mean()

train_writer = tf.summary.create_file_writer(log_path.joinpath('summary','train').as_posix())
validation_writer = tf.summary.create_file_writer(log_path.joinpath('summary','validation').as_posix())

with train_writer.as_default(0):
    config = FLAGS.GET_CONFIG()
    note = f'Note:  \n{FLAGS.LOGGING.NOTE}'
    tf.summary.text('Detail',[config,note])

step = 1
for e in range(total_epochs):
    for batch_data in train_data:
        image,bbox = batch_data
        with tf.GradientTape() as tape:
            raw_xywh = model(image,training=True)
            xywh = tf.keras.activations.sigmoid(raw_xywh)
            loss = GIOU_loss(bbox,xywh)
            train_metric.update_state(loss)
            loss = tf.math.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        step += 1
        if step%steps_per_log == 0:
            with train_writer.as_default(step):
                tf.summary.scalar('GIOU loss', train_metric.result())
                tf.summary.scalar('Learning rate', schedule(step))
            train_metric.reset_state()
    
    for batch_data in val_data:
        image,bbox = batch_data
        xywh = model.predict(image)
        loss = GIOU_loss(bbox,xywh)
        val_metric.update_state(loss)
    
    save_weights_path = log_path.joinpath('weights',f'{e:0>4}')
    save_weights_path.mkdir(parents=True)
    model.save_weights(save_weights_path.joinpath('weights').as_posix())
    
    current_GIOU_loss = val_metric.result()
    with validation_writer.as_default(step):
        tf.summary.scalar('GIOU loss', current_GIOU_loss)
    if e == 0:
        best_GIOU_loss = current_GIOU_loss
    else:
        if current_GIOU_loss < best_GIOU_loss:
            best_GIOU_loss = current_GIOU_loss
            with train_writer.as_default(step):
                context = '  \n'.join([f'Epoch: {e}',f'Step: {step}', f'Loss: {best_GIOU_loss}', f"Weight: {save_weights_path.joinpath('weights').as_posix()}"])
                tf.summary.text('Best GIOU loss',context)
    val_metric.reset_state()

    test_images, test_filepaths= next(test_data)
    test_preds = model.predict(test_images)
    print('='*20,f'EPOCH:{e:0>4}','='*20)
    for test_filepath,test_pred in zip(test_filepaths,test_preds):
        test_img = plt.imread(test_filepath.as_posix())/255
        utils.draw_bbox_on_image(test_img,test_pred,lw=10)