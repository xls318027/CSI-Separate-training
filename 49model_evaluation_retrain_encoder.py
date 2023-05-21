import os
import sys
import pandas as pd
from transformer_model_design import *
#from ue_encoder_model_design_trans import *
from tensorflow.python.keras.callbacks import EarlyStopping
import math
'''=============== You need to configure here: ====================================='''
# Set feedback_bits
feedback_bits =49
#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.Session(config=config)
tf.python.keras.backend.clear_session()  # 清理session


x_test = np.load('DATA/x_test.npy')

#x_train = np.load("p_x_train_1.npy")
_custom_objects = get_custom_objects()
my_objects = {'MultiHeadAttention': MultiHeadAttention,'cal_score_tf': cal_score_tf}
_custom_objects.update(my_objects)

custom_objects = get_custom_objects()


gnb_autoencoder_address = '49best_retrain_autoencoder.h5'
model_autoencoder = tf.keras.models.load_model(gnb_autoencoder_address, custom_objects=_custom_objects)

INPUT_SHAPE = 768

out1, out2, out3 = model_autoencoder.predict([x_test,x_test,x_test])
SGCS_1 = cal_score(x_test,out1,x_test.shape[0],12)
SGCS_2 = cal_score(x_test,out2,x_test.shape[0],12)
SGCS_3 = cal_score(x_test,out3,x_test.shape[0],12)
print('decoder1_test_SGSC is ' + np.str(SGCS_1))
print('decoder2_test_SGSC is ' + np.str(SGCS_2))
print('decoder3_test_SGSC is ' + np.str(SGCS_3))
