import os
import sys

import numpy as np
import pandas as pd
from transformer_model_design import *
from tensorflow.python.keras.callbacks import EarlyStopping

'''=============== You need to configure here: ====================================='''
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.Session(config=config)
tf.python.keras.backend.clear_session()  # 清理session
# latent space vectors acquisition
# Set the data set path and channel configurations
# 'data_dir' need to match the saved path of the downloaded data set


x_train_1 = np.load('DATA/x_train_1.npy')
x_train_2 = np.load('DATA/x_train_2.npy')
x_train_3 = np.load('DATA/x_train_3.npy')

x_val_1 = np.load('DATA/x_val_1.npy')
x_val_2 = np.load('DATA/x_val_2.npy')
x_val_3 = np.load('DATA/x_val_3.npy')

# encoder
_custom_objects = get_custom_objects()  # load keywords of Custom layers
my_objects = {'MultiHeadAttention': MultiHeadAttention, 'cal_score_tf': cal_score_tf}
_custom_objects.update(my_objects)

gnb_encoder_address1 = '49sep1_3v1_encoder_t6.h5'
model_encoder1 = tf.keras.models.load_model(gnb_encoder_address1, custom_objects=_custom_objects)
train_encode_feature1 = model_encoder1.predict(x_train_1)
val_encode_feature1 = model_encoder1.predict(x_val_1)

gnb_encoder_address2 = '49sep2_3v1_encoder_t5.h5'
model_encoder2 = tf.keras.models.load_model(gnb_encoder_address2, custom_objects=_custom_objects)
train_encode_feature2 = model_encoder2.predict(x_train_2)
val_encode_feature2 = model_encoder2.predict(x_val_2)

gnb_encoder_address3 = '49sep3_3v1_encoder_cnn.h5'
model_encoder3 = tf.keras.models.load_model(gnb_encoder_address3, custom_objects=_custom_objects)
train_encode_feature3 = model_encoder3.predict(x_train_3)
val_encode_feature3 = model_encoder3.predict(x_val_3)

train_feature = np.concatenate([train_encode_feature1,train_encode_feature2,train_encode_feature3],axis=0)
np.save("train_feature.npy",train_feature) # concatenated train labels

val_feature = np.concatenate([val_encode_feature1,val_encode_feature2,val_encode_feature3],axis=0)
np.save("val_feature.npy",val_feature)    # concatenated val labels
