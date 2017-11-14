import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784]) # 28pixel x 28pixel
Y = tf.placeholder(tf.float32, [None, 10]) # 10개의 숫자

# 784개의 입력에서 256개의 은닉층을 추출하고 또 다시 256개의 은닉층을 추출한 후,
# 10개의 결과값을 내놓는 신경망

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))
    L1 = tf.nn.dropout(L1, 0.8)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))
    L2 = tf.nn.dropout(L2, 0.8)

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01), name='W3')
    model = tf.matmul(L2, W3)

with tf.name_scope('output'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

tf.summary.scalar('cost', cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
keep_prob = tf.placeholder(tf.float32)

for epoch in range(30):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], # _,의 의미 : 코스트 value만 사용하겠다는 것.
                               feed_dict={X:batch_xs, Y:batch_ys, keep_prob:0.8})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost: ' '{:.3f}'.format(total_cost / total_batch))

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : ', sess.run(accuracy, feed_dict={X : mnist.test.images, Y: mnist.test.labels, keep_prob:1}))

labels = sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})
fig = plt.figure()

for i in range(10) :
    subplot = fig.add_subplot(2, 5, i + 1) # 2행 5열의 그래프를 만들고, i+1번째 숫자 출력.
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap=plt.cm.gray_r)

plt.show()

# saver = tf.train.Saver(tf.global_variables())
#
# ckpt = tf.train.get_checkpoint_state('C:\models_v1')
#
# if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#     saver.restore(sess, ckpt.model_checkpoint_path)
# else:
#     sess.run(tf.global_variables_initializer())