import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, (8 - B):]).reshape(-1,Num_.shape[1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 0]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)

    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)
    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)

    def custom_grad(dy):
        grad = dy
        return (grad, grad)

    return result, custom_grad


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B, **kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


@tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)

    return result, custom_grad


class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B, **kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


def Encoder_MLP(x, feedback_bits):
    B = 2

    def add_common_layers(y):
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.LeakyReLU()(y)
        return y

    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = add_common_layers(x)
    x = layers.Dense(units=int(feedback_bits / B), activation='sigmoid')(x)
    encoder_output = QuantizationLayer(B)(x)
    return encoder_output


def Decoder_cnn(x, feedback_bits):
    B = 2

    def add_common_layers(y):
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.LeakyReLU()(y)
        return y

    decoder_input = DeuantizationLayer(B)(x)
    x = tf.reshape(decoder_input,(-1, int(feedback_bits / B)))
    x = layers.Dense(768, activation='sigmoid')(x)
    x = layers.Reshape((12, 32, 2))(x)
    shortcut = x
    x = layers.Conv2D(128, (1, 3), padding='SAME', data_format='channels_last')(x)
    x = keras.layers.LeakyReLU()(x)

    for i in range(27):
        x = layers.Conv2D(128, kernel_size=(1, 3), padding='same', data_format='channels_last')(x)
        x = add_common_layers(x)
    x = layers.Conv2D(2, kernel_size=(1,3), padding='same', data_format='channels_last')(x)
    decoder_output = keras.layers.Add()([shortcut, x])
    decoder_output = layers.Flatten()(decoder_output)
    return decoder_output


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.H
    num1 = np.sqrt(vector_a * vector_a.H)
    num2 = np.sqrt(vector_b * vector_b.H)
    cos = (num / (num1 * num2))
    return cos


def cal_score(w_true, w_pre, NUM_SAMPLES, NUM_SUBBAND):
    img_total = 64
    num_sample_subband = NUM_SAMPLES * NUM_SUBBAND
    W_true = np.reshape(w_true, [num_sample_subband, img_total])
    W_pre = np.reshape(w_pre, [num_sample_subband, img_total])
    W_true2 = W_true[0:num_sample_subband, 0:int(img_total):2] + 1j * W_true[0:num_sample_subband, 1:int(img_total):2]
    W_pre2 = W_pre[0:num_sample_subband, 0:int(img_total):2] + 1j * W_pre[0:num_sample_subband, 1:int(img_total):2]
    score_cos = 0
    for i in range(num_sample_subband):
        W_true2_sample = W_true2[i:i + 1, ]
        W_pre2_sample = W_pre2[i:i + 1, ]
        score_tmp = cos_sim(W_true2_sample, W_pre2_sample)
        score_cos = score_cos + abs(score_tmp) * abs(score_tmp)
    score_cos = score_cos / num_sample_subband
    return score_cos


def cal_score_tf(w_true, w_pre, reduction='mean'):
    num_batch, num_sc, num_ant = w_true.shape[0], 12, 32
    w_true = tf.reshape(w_true, [-1, num_ant, 2])
    w_pre = tf.reshape(w_pre, [-1, num_ant, 2])

    w_true_re, w_true_im = w_true[..., 0], w_true[..., 1]
    w_pre_re, w_pre_im = w_pre[..., 0], w_pre[..., 1]

    numerator_re = tf.reduce_sum((w_true_re * w_pre_re + w_true_im * w_pre_im), -1)
    numerator_im = tf.reduce_sum((w_true_im * w_pre_re - w_true_re * w_pre_im), -1)
    denominator_0 = tf.reduce_sum((tf.square(w_true_re) + tf.square(w_true_im)), -1)
    denominator_1 = tf.reduce_sum((tf.square(w_pre_re) + tf.square(w_pre_im)), -1)
    cos_similarity = tf.sqrt(tf.square(numerator_re) + tf.square(numerator_im)) / (
                tf.sqrt(denominator_0) * tf.sqrt(denominator_1))
    cos_similarity = tf.square(cos_similarity)

    if reduction == 'mean':
        cos_similarity_scalar = tf.reduce_mean(cos_similarity)
    elif reduction == 'sum':
        cos_similarity_scalar = tf.reduce_sum(cos_similarity)

    return 1 - cos_similarity_scalar


def average_weights(weights1, weights2, alpha):
    weightsAve = []
    for i in range(len(weights1)):
        weights_merge = (1 - alpha) * weights1[i] + alpha * weights2[i]
        weightsAve.append(weights_merge)
    return weightsAve


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

# Return keywords of your own custom layers to ensure that model
# can be successfully loaded in test file.
def get_custom_objects():
    return {"QuantizationLayer": QuantizationLayer, "DeuantizationLayer": DeuantizationLayer}
