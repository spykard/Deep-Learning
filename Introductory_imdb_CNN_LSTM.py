'''
The classic and simple LSTM on keras example, from the official repository.
~~Train a recurrent convolutional network on the IMDB sentiment classification task.~~
~~Baseline: gets to 0.8498 test accuracy after 2 epochs. 41 s/epoch on K520 GPU.~~
~~batch_size is highly sensitive.~~
~~Only 2 epochs are needed as the dataset is very small.~~
'''

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import numpy as np
import random

## PARAMETERS ##
print_tensorflow_GPU_info = False
wandb.init(entity='unreal', project='Introductory_imdb_CNN_LSTM', name=None, notes=None, anonymous=None)
config = wandb.config  # config is a variable that holds and saves hyperparameters and inputs
config.random_state = 22
config.tensorflow_verbosity = "INFO"  # DEBUG(10): All | INFO(20): Info&Warning | WARN(30)[Default]: Warning | ERROR(40): Error | FATAL(50): None
    # Embedding
config.max_features = 20000
config.maxlen = 100
config.embedding_size = 128
config.dropout = 0.25
    # Convolution
config.kernel_size = 5
config.filters = 64
config.pool_size = 4
    # LSTM
config.lstm_output_size = 70
    # Training
config.batch_size = 30
config.epochs = 2
config.loss="binary_crossentropy"
config.optimizer="adam"
config.eval_metrics="accuracy"
##             ##

## Reproducibility ## 
random.seed(config.random_state)  # Python's seed
np.random.seed(config.random_state)  # Numpy's seed
tf.set_random_seed(config.random_state)  # Tensorflow's seed
##                 ##

## RTX GPU Memory BUG Fix & Must also be placed at the top of the code else it doesn't work ##
from keras.backend import tensorflow_backend as K
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True                     # dynamically grow the memory used on the GPU
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9  # fraction of the GPU to be used
#tf_config.log_device_placement = True                        # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=tf_config)
K.set_session(sess)                     # set this TensorFlow session as the default session for Keras
##                                                                                          ##     

## Tensorflow Verbosity Module ##
default_verbosity = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(config.tensorflow_verbosity)
print(f"\n--CHANGED TENSORFLOW VERBOSITY FROM {default_verbosity/10:.0f} (default) TO {tf.compat.v1.logging.get_verbosity()/10:.0f}")
##                             ##

## Tensorflow GPU Information Module ##
if print_tensorflow_GPU_info == True:
    print(f"\n--AVAILABLE GPUS:")
    K._get_available_gpus()
    print(f"\n--NUM OF GPUs AVAILABLE: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    print(f"\n--IS TF BUILT WITH CUDA: {tf.test.is_built_with_cuda()}")
    print(f"\n--IS GPU AVAILABLE: {tf.test.is_gpu_available()}")
##                                   ##  

# Data Loading and Preprocessing
print(f"\nLoading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=config.max_features)
print(f"{len(x_train)} train sequences")
print(f"{len(x_test)} test sequences")

print(f"Padding sequences (samples x time)")
x_train = sequence.pad_sequences(x_train, maxlen=config.maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=config.maxlen)
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")

# Model Building
print(f"\nBuilding model...")
model = Sequential()
model.add(Embedding(config.max_features, config.embedding_size, input_length=config.maxlen))
model.add(Dropout(config.dropout))
model.add(Conv1D(config.filters,
                 config.kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=config.pool_size))
model.add(LSTM(config.lstm_output_size))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=config.loss,
              optimizer=config.optimizer,
              metrics=[config.eval_metrics])

# Model Training
print(f"Training...")
model.fit(x_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(x_test, y_test),
          callbacks=[WandbCallback(monitor="val_loss", mode="auto", save_weights_only=False, save_model=False)])  # can also operate similarly to ModelCheckpoint as well as a validator for plotting

# Model Evaluation
score, acc = model.evaluate(x_test, y_test, batch_size=config.batch_size)
print(f"Test loss: {score*100:.5f}")
print(f"Test accuracy: {acc*100:.5f}")