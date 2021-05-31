# # 1 신경망 학습 구현하는 데 필요한 라이브러리 선언
# import numpy as np
# import matplotlib.pyplot as plt
#
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
#
#
# # 2 MNIST 데이터셋 불러오기
# (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
#
# # 3 데이터 전처리하기
# train_images = train_images.reshape((60000, 28, 28, 1))
# test_images = test_images.reshape((10000, 28, 28, 1))
# train_images, test_images = train_images / 255.0, test_images / 255.0
#
# # 4 합성곱 신경망 구축
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
#
# model.summary()
#
# # 5 Dense층 추가
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# model.summary()
#
# # 6 모델 컴파일
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# # 7 훈련
# model.fit(train_images, train_labels, epochs=5)
#
# # 모델 평가
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)

# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("학습셋 이미지 수 : %d 개" %(X_train.shape[0]))
print("테스트셋 이미지 수 : %d 개" %(X_test.shape[0]))
a = tf.get
from keras_explain.guided_bp import GuidedBP

explainer = GuidedBP(model)
exp = explainer.explain(image, target_class)