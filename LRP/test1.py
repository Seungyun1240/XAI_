# import sys
#
# from keras.datasets import mnist
# (X_train, y_class_train), (X_test, y_class_test) = mnist.load_data()
# print("학습셋 이미지 수 : %d 개" %(X_train.shape[0]))
# print("테스트셋 이미지 수 : %d 개" %(X_test.shape[0]))
#
# import matplotlib.pyplot as plt
# plt.imshow(X_train[0], cmap='gray')
# plt.show()
#
# for x in X_train[0]:
#     for i in x:
#         sys.stdout.write('%d' %i)
#     sys.stdout.write('\n')
#
# # 딥러닝에 필요한 케라스 함수 호출
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense
#
# # 필요 라이브러리 호출
# import numpy
# import tensorflow as tf
#
# # 데이터 셋 호출
# from keras.datasets import mnist
#
# # 실행 시마다 같은 결과값 도출을 위한 시드 설정
# numpy.random.seed(0)
# tf.random.set_seed(0)
#
# # 데이터를 불러와서 각 변수에 저장
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#
# # 학습에 적합한 형태로 데이터 가공
# X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
# X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255
#
# # 클래스를 학습에 이용하기 위해 데이터 가공
# Y_train = np_utils.to_categorical(Y_train, 10)
# Y_test = np_utils.to_categorical(Y_test, 10)
#
# # 딥러닝 모델 구조 설정(2개층, 512개의 뉴런 연결, 10개 클래스 출력 뉴런, 784개 픽셀 input 값, relu와 softmax 활성화 함수 이용)
# model = Sequential()
# model.add(Dense(512, input_dim=784, activation='relu'))
# model.add(Dense(10, activation='softmax'))
#
# # 딥러닝 구조 설정(loss 옵션을 다중 클래스에 적합한 categorical_crossentropy, 옵티마이저는 adam 설정)
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])
#
# # 모델 실행(X_test, Y_test로 검증, 200개씩 30번 학습)
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=2)
#
# # 학습 정확도, 검증 정확도 출력
# print('\nAccuracy: {:.4f}'.format(model.evaluate(X_train, Y_train)[1]))
# print('\nVal_Accuracy: {:.4f}'.format(model.evaluate(X_test, Y_test)[1]))

import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session

import math
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


class FilterViz():

    def __init__(self):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.session)
        self._model = load_model('models/pretrained/mnist_model.h5')

        self._img_rows = 28
        self._img_cols = 28
        (self._x_train, y_train), (self._x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            self._x_train = self._x_train.reshape(self._x_train.shape[0], 1, self._img_rows, self._img_cols)
            self._x_test = self._x_test.reshape(self._x_test.shape[0], 1, self._img_rows, self._img_cols)
        else:
            self._x_train = self._x_train.reshape(self._x_train.shape[0], self._img_rows, self._img_cols, 1)
            self._x_test = self._x_test.reshape(self._x_test.shape[0], self._img_rows, self._img_cols, 1)

        self._x_train = self._x_train.astype('float32')
        self._x_test = self._x_test.astype('float32')
        self._x_train /= 255
        self._x_test /= 255

    def _get_hidden_layers(self, data):
        with self.graph.as_default():
            set_session(self.session)
            feature_extractor = tf.keras.Model(inputs=self._model.inputs,
                                               outputs=[layer.output for layer in self._model.layers[:-4]])
            image = tf.convert_to_tensor(np.expand_dims(data, axis=0))
            result = feature_extractor(image)
        return result

    def _plot_filter(self, feature):
        feature = K.eval(feature)
        # filters = feature.shape[3]
        filters = 18
        plt.figure(1, figsize=(8, 15))
        n_columns = 3
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(0, filters):
            plt.subplot(n_rows, n_columns, i + 1)
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(feature[0, :, :, i], interpolation='nearest')
        img = BytesIO()
        plt.savefig(img, format='png', dpi=200)
        plt.clf()
        plt.cla()
        plt.close()
        img.seek(0)
        return img

    '''
    data: (28, 28)
    layer range: 0, 1, 2, 3
    '''

    def get_FilterViz(self, data, layer):
        data = data.reshape(self._img_rows, self._img_cols, 1)
        data = data.astype('float32')
        data /= 255

        hiddens = self._get_hidden_layers(data)
        return self._plot_filter(hiddens[layer])



import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.ops import gen_nn_ops
from tensorflow.keras.models import load_model
from io import BytesIO
import json
import base64


def getGradient(activation, weight, bias):
    W = tf.math.maximum(0., weight)
    b = tf.math.maximum(0., bias)
    z = tf.matmul(activation, W) + b

    dX = tf.matmul(1 / z, tf.transpose(W))

    return dX


def backprop_dense(activation, weight, bias, relevance):
    W = tf.math.maximum(0., weight)
    b = tf.math.maximum(0., bias)
    z = tf.matmul(activation, W) + b

    s = relevance / (z + 1e-10)
    c = tf.matmul(s, tf.transpose(W))

    return activation * c


def backprop_pooling(activation, relevance):
    z = MaxPool2D(pool_size=(2, 2))(activation)

    s = relevance / (z + 1e-10)
    c = gen_nn_ops.max_pool_grad_v2(orig_input=activation, orig_output=z, grad=s,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='VALID')

    return activation * c


def backprop_conv(activation, weight, bias, relevance):
    strides = (1, 1)
    W = tf.math.maximum(0., weight)
    b = tf.math.maximum(0., bias)

    layer = Conv2D(filters=W.shape[-1], kernel_size=(W.shape[0], W.shape[1]),
                   padding='VALID', activation='relu')

    layer.build(input_shape=activation.shape)

    layer.set_weights([W, b])

    z = layer(activation)

    s = relevance / (z + 1e-10)

    c = tf.compat.v1.nn.conv2d_backprop_input(activation.shape, W, s, [1, *strides, 1], padding='VALID')

    return activation * c


# data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 221, 251, 255, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 182, 237, 255, 255, 255, 250, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 244, 255, 255, 239, 179, 94, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 144, 255, 255, 204, 0, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 218, 255, 219, 0, 33, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 246, 255, 116, 66, 251, 255, 255, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 255, 60, 254, 255, 255, 248, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 255, 253, 255, 246, 255, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 221, 255, 255, 255, 173, 255, 230, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 232, 128, 97, 255, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 138, 255, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 194, 255, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 255, 113, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 255, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 255, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 255, 255, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 255, 213, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 255, 169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 166, 255, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 234, 255, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 255, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 255, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 255, 251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 255, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 255, 246, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 115, 255, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 147, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
data = json.loads(input())

data = np.array(data)

data_shape = (1, 28, 28, 1)
data = data.astype('float32')
data /= 255
data_seg = tf.cast(data.reshape(data_shape), tf.float32)

model = load_model('models/pretrained/mnist_model.h5')
output_shapes = []
layers = []
weights = model.get_weights()[::2]
biases = model.get_weights()[1::2]

for e, i in enumerate(model.layers):
    if (("Conv" in str(i)) or ("MaxPooling" in str(i))
            or ("Flatten" in str(i)) or ("Dense" in str(i))):
        output_shapes.append(i.output_shape[1:])
        layers.append(i)

activations = []

a = data_seg
for layer in layers:
    a = layer(a)
    activations.append(a)

R = activations[-1]
wb_cnt = 0

for layer_num, layer in enumerate(layers[::-1]):

    if ("Flatten" in str(layer)):
        R = tf.reshape(R, (-1, *output_shapes[(~layer_num) - 1]))
    elif ("Dense" in str(layer)):
        a = activations[(~layer_num) - 1] if layer_num != (len(layers) - 1) else data_seg
        w = weights[~wb_cnt]
        b = biases[~wb_cnt]
        R = backprop_dense(a, w, b, R)
        wb_cnt += 1
    elif ("Conv" in str(layer)):
        a = activations[(~layer_num) - 1] if layer_num != (len(layers) - 1) else data_seg
        w = weights[~wb_cnt]
        b = biases[~wb_cnt]
        R = backprop_conv(a, w, b, R)
        wb_cnt += 1
    elif ("MaxPooling" in str(layer)):
        a = activations[(~layer_num) - 1] if layer_num != (len(layers) - 1) else data_seg
        R = backprop_pooling(a, R)

LRP_out = tf.reshape(tf.reduce_sum(R, axis=-1), data_shape[1:-1])

plt.imshow(LRP_out, cmap=plt.cm.jet)
plt.axis('off')
plt.tight_layout()

img = BytesIO()
plt.savefig(img, format='png', dpi=200)
plt.clf()
plt.cla()
plt.close()
img.seek(0)

print(base64.b64encode(img.getvalue()).decode(), end='')


