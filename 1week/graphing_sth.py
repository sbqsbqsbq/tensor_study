import tensorflow as tf

# 1 x 2 Matrix Generation
# This Operation will join default graph as a node

matrix1 = tf.constant([[3., 3.]]) # 1 x 2 Matrix
matrix2 = tf.constant([[2.], [2.]]) # 2 x 1 Matrix

product = tf.matmul(matrix1, matrix2)  # (3, 3) * (2)   = (12)
                                       #          (2)

sess = tf.Session() # Call a Session
result = sess.run(product) # run 함수는 작업의 실행결과를 알려주는 함수임.
print(result)  # [[ 12.]]

sess.close() # 세션은 반드시 닫아줘야 함.

# 시스템 자원을 쉽게 관리하려면 with 함수를 사용하면 됨.
# Example)
# with tf.Session() as sess:
#   result = sess.run([product])
#   print(result)

# 연산에 GPU를 활용할 때도 with 문을 사용함.
# Example)
# with tf.Session() as sess:
#   with tf.device("/gpu:1"):
#       matrix1 = tf.constant([[3., 3.]])
#       matrix2 = tf.constant([[2.], [2.]])
#       product = tf.matmul(matrix1, matrix2)
