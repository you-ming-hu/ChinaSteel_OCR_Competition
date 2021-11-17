from easydict import EasyDict as edict

DATA = edict()
DATA.TRAIN = edict()
DATA.TRAIN.TABLE_PATH = None
DATA.TRAIN.DROP_BAD_BBOX_DATA = None
DATA.TRAIN.VALIDATION_SPLIT_RATIO = None
DATA.TRAIN.VALIDATION_SPLIT_RANDOM_STATE = None
DATA.TRAIN.IMAGE_PATH = None
DATA.TRAIN.TRAIN_BATCH_SIZE = None
DATA.TRAIN.VAL_BATCH_SIZE = None
DATA.TEST = edict()
DATA.TEST.IMAGE_PATH = None
DATA.TEST.BATCH_SIZE = None

MODEL = None

OPTIMIZER = edict()
OPTIMIZER.TYPE = None
OPTIMIZER.MAX_LEARNING_RATE = None
OPTIMIZER.SCHEDULE_GAMMA = None

EPOCHS = edict()
EPOCHS.TOTAL = None
EPOCHS.WARMUP = None

LOGGING = edict()
LOGGING.PATH = None
LOGGING.MODEL_NAME = None
LOGGING.TRIAL_NUMBER = None
LOGGING.NOTE = None
LOGGING.SAMPLES_PER_LOG = None
LOGGING.TEST_IMAGE_COLUMNS = None

def GET_CONFIG():
    return '  \n'.join([
        'Config:',
        f'{DATA.TRAIN.DROP_BAD_BBOX_DATA = }',
        f'{DATA.TRAIN.VALIDATION_SPLIT_RATIO = }',
        f'{DATA.TRAIN.VALIDATION_SPLIT_RANDOM_STATE = }',
        f'{DATA.TRAIN.TRAIN_BATCH_SIZE = }',
        f'OPTIMIZER.TYPE = {OPTIMIZER.TYPE.__name__}',
        f'{OPTIMIZER.MAX_LEARNING_RATE = }',
        f'{OPTIMIZER.SCHEDULE_GAMMA = }',
        f'{EPOCHS.TOTAL = }',
        f'{EPOCHS.WARMUP = }',
        f'{LOGGING.SAMPLES_PER_LOG = }'
        ])
    
def CHECK():
    assert isinstance(DATA.TRAIN.TABLE_PATH,str)
    assert isinstance(DATA.TRAIN.DROP_BAD_BBOX_DATA,bool)
    assert isinstance(DATA.TRAIN.VALIDATION_SPLIT_RATIO,float)
    assert isinstance(DATA.TRAIN.VALIDATION_SPLIT_RANDOM_STATE,int)
    assert isinstance(DATA.TRAIN.IMAGE_PATH,str)
    assert isinstance(DATA.TRAIN.TRAIN_BATCH_SIZE,int)
    assert isinstance(DATA.TRAIN.VAL_BATCH_SIZE,int)
    assert isinstance(DATA.TEST.IMAGE_PATH,str)
    assert isinstance(DATA.TEST.BATCH_SIZE,int)
    assert isinstance(OPTIMIZER.MAX_LEARNING_RATE,float)
    assert isinstance(OPTIMIZER.SCHEDULE_GAMMA,(float,int))
    assert isinstance(EPOCHS.TOTAL,int)
    assert isinstance(EPOCHS.WARMUP,int)
    assert isinstance(LOGGING.PATH,str)
    assert isinstance(LOGGING.MODEL_NAME,(int,str))
    assert isinstance(LOGGING.TRIAL_NUMBER,(int,str))
    assert isinstance(LOGGING.NOTE,str)
    assert isinstance(LOGGING.SAMPLES_PER_LOG,int)
    assert isinstance(LOGGING.TEST_IMAGE_COLUMNS,int)