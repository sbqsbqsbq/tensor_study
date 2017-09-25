from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession() # Interactive Session Initialize

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_:batch[1]})

# Training a model.

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Accuracy is about 91%
print(accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

# However, It's not good for getting 91% accuracy in neural network model. We need to
# increase accuracy.

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution 합성곱
# Pooling 풀링

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_poop_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# First convolution layer
# 5x5 윈도우에 32개의 필터를 가짐.

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# X를 4D 텐서로 reshape 하는 과정.
# 두 번째와 세 번째 차원은 이미지의 가로 세로 길이
# 마지막 차원은 컬러 채널의 수를 나타냄.

x_image = tf.reshape(x, [-1, 28, 28, 1])

# x_image와 가중치 텐서에 합성곱(convolution)을 적용
# ReLU 함수를 적용함.
# 출력값을 구하기 위해 max_pooling을 진행

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_poop_2x2(h_conv1)

# Second Convolution Layer
# 여기서는 두 번째 합성곱 계층이 5x5 윈도우에
# 64개 필터를 가짐.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_poop_2x2(h_conv2)

# 완전 연결 계층 (Fully Connected Layer)
# 이미지 크기를 7x7로 줄임.
# 1024개의 뉴런으로 연결되는 완전 연결 계층 구성.

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout을 적용함.
# placeholder를 만들어서, 테스트 과정에서 dropout이 적용되지 않도록 함.

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Final Softmax Layer
# 단일 계층 소프트맥스 회귀 모형을 만듦.

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Model training and Evaluation

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_:batch[1], keep_prob: 1.0
        })
        print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={
            x:batch[0], y_:batch[1], keep_prob: 0.5
        })

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
}))

