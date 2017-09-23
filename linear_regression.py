import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32) # 32 FLOAT
y_data = x_data * 0.1 + 0.3

# 1차 선형 회귀분석 시작
# y = W(coefficient) * x + b(error)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = W * x_data + b

# MSE(평균제곱잔차) 최소화
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 개시자(Initializer) 생성
init = tf.global_variables_initializer()

# 그래프 그리기
sess = tf.Session()
sess.run(init)

# 그래프 조정
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# W의 추정치는 0.1, b는 0.3

# 세션 종료 해줘야 함
sess.close()