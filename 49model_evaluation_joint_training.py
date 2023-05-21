import os
import sys
import pandas as pd
from transformer_model_design import *
from tensorflow.python.keras.callbacks import EarlyStopping

#Take joint_49bits_3v1 for example
'''=============== You need to configure here: ====================================='''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.Session(config=config)
tf.python.keras.backend.clear_session()  # clear session




x_test = np.load('DATA/x_test.npy')


# encoder
_custom_objects = get_custom_objects()  # load keywords of Custom layers
my_objects = {'MultiHeadAttention': MultiHeadAttention, 'cal_score_tf': cal_score_tf}
_custom_objects.update(my_objects)

gnb_encoder_address1 = '49sep1_3v1_encoder_t6.h5'
model_encoder1 = tf.keras.models.load_model(gnb_encoder_address1, custom_objects=_custom_objects)
en1 = model_encoder1.predict(x_test)


# xxx = QuantizationLayer(2)(encode_feature)
# Dq_encode_feature = xxx.numpy()
# decode_input = Dq_encode_feature



decoder_address_de = '49sep1_3v1_decoder_t6.h5'     # gNB_decoder_2
# _custom_objects_de = get_custom_objects()  # load keywords of Custom layers
model_decoder = tf.keras.models.load_model(decoder_address_de, custom_objects=_custom_objects)
y_test1 = model_decoder.predict(en1)
SGSC1 = cal_score(x_test, y_test1, x_test.shape[0], 12)
print('SGCS1 is ' + np.str(SGSC1))

