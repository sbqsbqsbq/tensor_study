import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = input1 * input2

# tf.placeholder()를 사용해서 특정 작업을 feed 받을 수 있게 함.
# 즉, graph 연산에게 직접 값을 입력함.

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))