import os

import keras
import pandas as pd
import numpy as np
from transformer_model_design import *
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard, Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from MLP_CNN_model_design import *
'''=============== You need to configure here: ====================================='''
# Set feedback_bits
feedback_bits = 49

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.Session(config=config)
tf.python.keras.backend.clear_session()  # 清理session



_custom_objects = get_custom_objects()
my_objects = {'MultiHeadAttention': MultiHeadAttention,'cal_score_tf': cal_score_tf,'QuantizationLayer':QuantizationLayer}
_custom_objects.update(my_objects)

x_train_1 = np.load('DATA/x_train_1.npy')
x_train_2 = np.load('DATA/x_train_2.npy')
x_train_3 = np.load('DATA/x_train_3.npy')

x_val_1 = np.load('DATA/x_val_1.npy')
x_val_2 = np.load('DATA/x_val_2.npy')
x_val_3 = np.load('DATA/x_val_3.npy')


#####


decoder_address_de_1 = '49sep1_3v1_decoder_t6.h5'
#model_decoder_1 = tf.keras.models.load_model(decoder_address_de_1, custom_objects=_custom_objects)

decoder_address_de_2 = '49sep2_1v3_decoder_t5.h5'
#model_decoder_2 = tf.keras.models.load_model(decoder_address_de_2, custom_objects=_custom_objects)

decoder_address_de_3 = '49sep3_1v3_decoder_cnn.h5'




EMBEDDING_DIM = 64 * 6
NUM_QUAN_BITS = 2
NUM_HEAD = 8
INPUT_SHAPE = 768  # 768

encoderInput = keras.Input(shape=(INPUT_SHAPE))
encoderOutput = Encoder(encoderInput, feedback_bits, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD)
encoder = keras.Model(inputs=encoderInput, outputs=encoderOutput, name='Encoder')
print("encoder summary")
encoder.summary()


decoder1 = tf.keras.models.load_model(decoder_address_de_1, custom_objects=_custom_objects)
for layer in decoder1.layers:
    layer.trainable = False
decoder1._name = "Decoder1"
print("decoder1 summary")
decoder1.summary()


decoder2 = tf.keras.models.load_model(decoder_address_de_2, custom_objects=_custom_objects)
for layer in decoder2.layers:
    layer.trainable = False
decoder2._name = "Decoder2"
print("decoder2 summary")
decoder2.summary()

decoder3 = tf.keras.models.load_model(decoder_address_de_3, custom_objects=_custom_objects)
for layer in decoder3.layers:
    layer.trainable = False
decoder3._name = "Decoder3"
print("decoder3 summary")
decoder3.summary()

autoencoderInput_1 = keras.Input(shape=(INPUT_SHAPE))
autoencoderInput_2 = keras.Input(shape=(INPUT_SHAPE))
autoencoderInput_3 = keras.Input(shape=(INPUT_SHAPE))
autoencoderOutput1 = decoder1(encoder(autoencoderInput_1))
autoencoderOutput2 = decoder2(encoder(autoencoderInput_2))
autoencoderOutput3 = decoder3(encoder(autoencoderInput_3))

#autoencoder_name = "fine_best_130bits_retrain_autoencoder.h5"

autoencoderModel = Model(inputs=[autoencoderInput_1,autoencoderInput_2,autoencoderInput_3], outputs=[autoencoderOutput1,autoencoderOutput2,autoencoderOutput3], name='Autoencoder')
#autoencoderModel = tf.keras.models.load_model(autoencoder_name,custom_objects=_custom_objects)
autoencoderModel.summary()



criterion = cal_score_tf
autoencoderModel.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss={'Decoder1':criterion,'Decoder2':criterion,'Decoder3':criterion},loss_weights={'Decoder1': 1/3, 'Decoder2': 1/3,'Decoder3': 1/3})



# Model training

def scheduler(epoch, lr):
    print(lr)
    if (epoch + 1) % 50 == 0:
        return max(lr * 0.5, 1e-5)
    else:
        return lr


##
checkpointer = ModelCheckpoint(filepath='noad_best_49_retrain_autoencoder.h5', verbose=1, save_best_only=True)
scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
earlystop_callback = EarlyStopping(monitor='val_loss', patience=50)
#gnb_encoder_address = 'from_30_best_242bits_retrain_encoder.h5'
#autoencoderModel = tf.keras.models.load_model(gnb_encoder_address, custom_objects=_custom_objects)
autoencoderModel.fit(x=[x_train_1,x_train_2,x_train_3], y=[x_train_1,x_train_2,x_train_3], batch_size=64, epochs=200, callbacks=[earlystop_callback, scheduler,checkpointer], verbose=2,
            validation_data=([x_val_1,x_val_2,x_val_3],[x_val_1,x_val_2,x_val_3]))



