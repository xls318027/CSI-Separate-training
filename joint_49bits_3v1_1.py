import os
import pandas as pd
import numpy as np
from transformer_model_design import *
from MLP_CNN_model_design import *
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard, Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
'''=============== You need to configure here: ====================================='''
# Set feedback_bits
feedback_bits = 49 # 49 87 130 142
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.compat.v1.Session(config=config)
tf.python.keras.backend.clear_session()  # clear session

x_train_1 = np.load('DATA/x_train_1.npy')
#x_train_2 = np.load('x_train_2.npy')
#x_train_3 = np.load('x_train_3.npy')

x_val_1 = np.load('DATA/x_val_1.npy')
#x_val_2 = np.load('x_val_2.npy')
#x_val_3 = np.load('x_val_3.npy')

# Each sample includes 768 real numbers
INPUT_SHAPE = 768  # CSI Eigenvectors dimension equals N_t * S * 2= 32 * 12 * 2 = 768


EMBEDDING_DIM = 64 * 6 # e_dim for transformer embedding dimension
NUM_QUAN_BITS = 2 # uniform quantization bits
NUM_HEAD = 8 # attention heads for multi-head attention


# encoder1: 6-layer transformer
encoderInput = keras.Input(shape=(INPUT_SHAPE))
encoderOutput = Encoder(encoderInput, feedback_bits, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD)
encoder = keras.Model(inputs=encoderInput, outputs=encoderOutput, name='encoder')

# decoder: 6-layer transformer
decoderInput = keras.Input(shape=(feedback_bits))
decoderOutput = Decoder(decoderInput, feedback_bits, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD)
decoder = keras.Model(inputs=decoderInput, outputs=decoderOutput, name="Decoder")

# joint training autoencoder model design
autoencoderInput = keras.Input(shape=(INPUT_SHAPE))
autoencoderOutput = decoder(encoder(autoencoderInput))

autoencoderModel = Model(inputs=autoencoderInput,
                         outputs=autoencoderOutput, name='Autoencoder')
autoencoderModel.summary()

criterion = cal_score_tf # loss function = 1- SCS
autoencoderModel.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=criterion)

# learning rate scheduler
def scheduler(epoch, lr):
    print(lr)
    if (epoch + 1) % 40 == 0:
        return max(lr * 0.5, 1e-5)
    else:
        return lr

## Some Callback Function define
checkpointer = ModelCheckpoint(filepath='best_49sep1_3v1_autoencoder.h5', verbose=1, save_best_only=True)
scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
earlystop_callback = EarlyStopping(monitor='val_loss', patience=50)

#model training
autoencoderModel.fit(x=x_train_1, y=x_train_1, batch_size=64, epochs=200,
                     callbacks=[earlystop_callback, scheduler,checkpointer],
                     verbose=2, validation_data=(x_val_1,x_val_1))
# Save models of encoder and decoder
encoder.save('49sep1_3v1_encoder_t6.h5')
decoder.save('49sep1_3v1_decoder_t6.h5')