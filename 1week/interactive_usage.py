import tensorflow as tf
sess= tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# x 변수가 run()이라는 함수를 사용할 수 있게 함.
x.initializer.run()

# x에서 a를 빼는 작업을 추가
sub = tf.subtract(x, a)
print(sub.eval())

# Session 종료
sess.close()