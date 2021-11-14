import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from sklearn import model_selection

import FLAGS
import Dataset

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

label_smoothing = FLAGS.LOSS.LABEL_SMOOTHING
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=label_smoothing)

bce_metric = tf.keras.metrics.BinaryCrossentropy(from_logits=False, label_smoothing=label_smoothing)
accuracy_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

train_writer = tf.summary.create_file_writer(log_path.joinpath('summary','train').as_posix())
validation_writer = tf.summary.create_file_writer(log_path.joinpath('summary','validation').as_posix())

with train_writer.as_default():
    config = [
        f'FLAGS.DATA.TRAIN.VALIDATION_SPLIT_RATIO = {FLAGS.DATA.TRAIN.VALIDATION_SPLIT_RATIO}',
        f'FLAGS.DATA.TRAIN.VALIDATION_SPLIT_RANDOM_STATE = {FLAGS.DATA.TRAIN.VALIDATION_SPLIT_RANDOM_STATE}',
        f'FLAGS.DATA.TRAIN.TRAIN_BATCH_SIZE = {FLAGS.DATA.TRAIN.TRAIN_BATCH_SIZE}',
        f'FLAGS.LOSS.LABEL_SMOOTHING = {FLAGS.LOSS.LABEL_SMOOTHING}'
        f'FLAGS.OPTIMIZER.TYPE = {FLAGS.OPTIMIZER.TYPE}',
        f'FLAGS.OPTIMIZER.MAX_LEARNING_RATE = {FLAGS.OPTIMIZER.MAX_LEARNING_RATE}',
        f'FLAGS.OPTIMIZER.SCHEDULE_GAMMA = {FLAGS.OPTIMIZER.SCHEDULE_GAMMA}',
        f'FLAGS.EPOCHS.TOTAL = {FLAGS.EPOCHS.TOTAL}',
        f'FLAGS.EPOCHS.WARMUP = {FLAGS.EPOCHS.WARMUP}'
        ]
    config = '  \n'.join(config)
    note = f'Note:  \n{FLAGS.LOGGING.NOTE}'
    tf.summary.test('Detail',[config,note])

step = 1
for e in range(total_epochs):
    for batch_data in train_data:
        image,overturn_label = batch_data
        with tf.GradientTape() as tape:
            predict = model(image)
            loss = bce_loss(overturn_label,predict)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        bce_metric.update_state(overturn_label,predict)
        accuracy_metric.update_state(overturn_label,predict)
        
        step += 1
        if step%steps_per_log == 0:
            with train_writer.as_default(step):
                tf.summary.scalar('BCE loss', bce_metric.result())
                tf.summary.scalar('Accuracy', accuracy_metric.result())
        
    bce_metric.reset_state()
    accuracy_metric.reset_state()
    
    print('='*20,f'EPOCH: {e}','='*20)
    for batch_data in validation_data:
        image,overturn_label = batch_data
        predict = model(image)
        loss = bce_loss(overturn_label,predict)
        bce_metric.update_state(overturn_label,predict)
        accuracy_metric.update_state(overturn_label,predict)

    save_weights_path = log_path.joinpath('weights',f'{e:0>4}')
    save_weights_path.mkdir()
    model.save_weights(save_weights_path.joinpath('weights').as_posix())

    current_BCE_loss = bce_metric.result()
    current_Accuracy = accuracy_metric.result()
    with validation_writer.as_default(step):
        tf.summary.scalar('BCE loss', current_BCE_loss)
        tf.summary.scalar('Accuracy', current_Accuracy)
        if e == 0:
            best_BCE_loss = current_BCE_loss
            best_Accuracy = current_Accuracy
        else:
            if current_BCE_loss < best_BCE_loss:
                best_BCE_loss = current_BCE_loss
                with train_writer.as_default(step):
                    context = '  \n'.join([f'Epoch: {e}',f'Step: {step}', f'Loss: {best_BCE_loss}', f"Weight: {save_weights_path.joinpath('weights').as_posix()}"])
                    tf.summary.text('Best BCE loss',context)
            if current_Accuracy < best_Accuracy:
                best_Accuracy = current_Accuracy
                with train_writer.as_default(step):
                    context = '  \n'.join([f'Epoch: {e}',f'Step: {step}', f'Loss: {best_Accuracy}', f"Weight: {save_weights_path.joinpath('weights').as_posix()}"])
                    tf.summary.text('Best Accuracy loss',context)
        
    bce_metric.reset_state()
    accuracy_metric.reset_state()
    
    test_imgs,_ = next(test_data)
    pred = model.predict(test_imgs)
    for img,p in zip(test_imgs,pred):
        print(p)
        plt.imshow(img,cmap='gray')
        plt.show()