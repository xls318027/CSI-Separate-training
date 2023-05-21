import os
import pandas as pd
import numpy as np
from transformer_model_design import *
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard, Callback
from tensorflow.keras.callbacks import ModelCheckpoint
'''=============== You need to configure here: ====================================='''
# Set feedback_bits
feedback_bits = 49 # 49 87 130 142
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.Session(config=config)
tf.python.keras.backend.clear_session()  # clear session



train_encode_feature = np.load("train_feature.npy")
val_encode_feature = np.load("val_feature.npy")
x_train_all = np.load("DATA/x_train_all.npy")
x_val_all = np.load("DATA/x_val_all.npy")
EMBEDDING_DIM = 64 * 6
NUM_QUAN_BITS = 2
NUM_HEAD = 8



INPUT_SHAPE = 768  # 768
decoderInput = keras.Input(shape=(feedback_bits))
decoderOutput = Decoder(decoderInput, feedback_bits, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD)
decoder = keras.Model(inputs=decoderInput, outputs=decoderOutput, name="Decoder")

criterion = cal_score_tf
decoder.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=criterion)
print(decoder.summary())




def scheduler(epoch, lr):
    print(lr)
    if (epoch + 1) % 50 == 0:
        return max(lr * 0.5, 3e-5)
    else:
        return lr


## Some Callback Function define
checkpointer = ModelCheckpoint(filepath='49_sep_3v1_retrain_decoder.h5', verbose=1, save_best_only=True)
scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
earlystop_callback = EarlyStopping(monitor='val_loss', patience=50)

decoder.fit(x=train_encode_feature, y=x_train_all, batch_size=64, epochs=100, callbacks=[earlystop_callback, scheduler,checkpointer], verbose=2,
            validation_data=(val_encode_feature,x_val_all))
# Save models of retrained_decoder
decoder.save('49bits_retrain_dedcoder.h5')
