import tensorflow as tf
import numpy as np

data = np.loadtxt('C:\data.csv', delimiter=',', unpack=True, dtype='float32')
# loadtext unpack을 해주면, 행렬을 해줌.

x_data = np.transpose(data[0:2])
# 불러온 상태에서 transpose를 하면 행렬을 바꿔줌.

y_data = np.transpose(data[2:])


global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1. ,1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.nn.relu(tf.matmul(L2, W3))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

# 세션 열고, 모델 불러들이고 저장
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
# tf.train.Server : 저장과 관련된 함수
# tf.global_vairables() : 앞서 저장된 모든 변수를 가져옴.

ckpt = tf.train.get_checkpoint_state('C:\model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

# tf.train."S"aver Saver는 반드시 대문자로 시작.
# tf.train."S"aver.'s'ave
# tf.train."S"aver.'r'estore

for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    print('Step: %d, ' % sess.run(global_step), 'Cost : %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

saver.save(sess, 'C:\model\dnn.ckpt', global_step=global_step)

prediction = tf.argmax(model,1)
target = tf.argmax(Y, 1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))