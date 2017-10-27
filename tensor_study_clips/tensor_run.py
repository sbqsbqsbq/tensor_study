import tensorflow as tf # 텐서플로우 임포트
hello = tf.constant("Hello, World!") # hello라는 String constant
print(hello) # 출력 ; Tensor("Const:0", shape=(), dtype=string)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b) # a + b로도 쓸 수 있음.
print(c) # 출력 : Tensor("Add:0", shape=(), dtype=int32) Graph만 그려둔 것.

sess = tf.Session()

print(sess.run([a,b,c])) # 결과 출력 [10, 32, 42] Rank : 1 Shape: 3
print(sess.run(hello)) # 결과 출력 Hello, World!