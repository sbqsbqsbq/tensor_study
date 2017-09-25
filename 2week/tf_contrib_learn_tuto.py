# TensorFlow Contribute Learning
# 텐서플로우 고수준 머신러닝
# 신경망 분류기를 생성
# Iris 데이터셋에 있는 꽃받침과 꽃잎의 정보를 이용하여
# 꽃의 종류를 예측함.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# DataSet
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = 'iris_test.csv'

# Loading data
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# 모든 특성이 실수(Real Number)라고 가정.
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# 10, 20, 10개 유닛을 가진 3층 DNN을 생성
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

# 모델을 학습 시킴.
classifier.fit(x=training_set.data,
              y=training_set.target,
              steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float
)

y = list(classifier.predict(new_samples, as_iterable=True))
print('Prediction: {}'.format(str(y)))

# 1번째 표본을 Iris versicolor로,
# 2번째 표본을 Iris versicolor로 인식함.