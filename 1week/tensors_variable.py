import tensorflow as tf

state = tf.Variable(0, name="counter")

# state에 1을 더하는 작업을 만들자.

one = tf.constant(1)
new_value = tf.add(state, one) # New Operation 새 작업
update = tf.assign(state, new_value) # state에 새 값을 할당(assign)함

# 그래프를 한 번 작동시킨 후에는 init을 통해서, 작업을 초기화 해야 함.
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op) # init작업 실행
    print(sess.run(state)) # state 값을 업데이트 하고 출력하는 작업 시작
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))