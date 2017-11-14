# 텐서플로우 임포트
import tensorflow as tf
# hello라는 String constant
hello = tf.constant("Hello, World!")
# 출력 : Tensor("Const:0", shape=(), dtype=string)
print(hello)

a = tf.constant(10)
b = tf.constant(32)

# a + b로도 쓸 수 있음.
c = tf.add(a, b)

# 출력 : Tensor("Add:0", shape=(), dtype=int32) Graph만 그려둔 것.
print(c)

sess = tf.Session()

# 결과 출력 [10, 32, 42] Rank : 1 Shape: 3
# 결과 출력 Hello, World!
print(sess.run([a,b,c]))
print(sess.run(hello))

# 플레이스 홀더에 Float 변수형 데이터 미리 저장.
# 텐서플로우에서의
# None은 정해지지 않았다는 뜻
# 출력은 Tensor("Placeholder:0", shape=(?, 3), dtype=float32)
X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1, 2, 3], [4, 5, 6]]
y_data = [2, 4, 6]

W = tf.Variable(tf.random_normal([3, 2])) # getVariable 이란 메소드가 있음.
b = tf.Variable(tf.random_normal([3, 1]))

expr = tf.matmul(X, W) + b # Y = W*X + b
sess.run(tf.global_variables_initializer())

print("### X_DATA ###")
print(x_data)
print("### W DATA ###")
sess.run(W)
print("### b DATA ###")
sess.run(b)
print("### expr ###")
sess.run(expr)