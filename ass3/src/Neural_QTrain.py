import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 100000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy



# TODO: HyperParameters
GAMMA = 0.9 # discount factor
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period

LR = 0.0001
memory_size = 10000
batch_size = 32
replace_target_iter = 30
learn_step_counter = 0
memory_counter = 0
# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --


# TODO: Define Network Graph
memory = np.zeros((memory_size,STATE_DIM*2+2))
# print(type(memory))
# state_in = tf.placeholder(tf.float32, [None, STATE_DIM])
# next_state = tf.placeholder(tf.float32,[None,STATE_DIM])
# action_in = tf.placeholder(tf.int32, [None,])
# reward = tf.placeholder(tf.float32, [None,])
state_in = tf.placeholder(tf.float32, [None, STATE_DIM], name='state_in')  # input State
reward = tf.placeholder(tf.float32, [None, ], name='reward')  # input Reward
action_in = tf.placeholder(tf.int32, [None, ], name='action_in')  # input Action
next_state = tf.placeholder(tf.float32, [None,STATE_DIM])
# target_in = tf.placeholder("float", [None])
# q_target = tf.placeholder(tf.float32,[None,ACTION_DIM],name='Q_target')
w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

with tf.variable_scope('eval_net'):
    e1 = tf.layers.dense(state_in, 20, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e1')
    q_values = tf.layers.dense(e1, ACTION_DIM, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='q')



# ------------ build target network ------------------
with tf.variable_scope('target_net'):
    t1 = tf.layers.dense(next_state, 20, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='t1')
    q_next = tf.layers.dense(t1, ACTION_DIM, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='t2')

with tf.variable_scope('q_target'):
    q_target = reward + GAMMA * tf.reduce_max(q_next, axis=1, name='Qmax_s_')    # shape=(None, )
    q_target = tf.stop_gradient(q_target)
with tf.variable_scope('q_values'):
    a_indices = tf.stack([tf.range(tf.shape(action_in)[0], dtype=tf.int32), action_in], axis=1)
    q_eval_wrt_a = tf.gather_nd(params=q_values, indices=a_indices)    # shape=(None, )
with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval_wrt_a, name='TD_error'))
with tf.variable_scope('train'):
    train_op = tf.train.RMSPropOptimizer(LR).minimize(loss)

#--- 替换 target net 参数 -----
t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
with tf.variable_scope('soft_replacement'):
    target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

def store_transition(state_in,action,reward,s_): 
    global memory_counter  
    # if('memory_counter' in globals()):
    transition = np.hstack((state_in, [action,reward],s_))
    index = memory_counter % memory_size
    # print(([action,reward]))
    # print(transition)
    # print("State In")
    # print(state_in)
    # print("Next_state")
    # print(next_state)
    memory[index, :] = transition
    # print(memory[0])
    memory_counter+=1
        



            




# # TODO: Network outputs
# q_values = 
# q_action =

# # TODO: Loss/Optimizer Definition


# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

def choose_action(state,epsilon):
        # to have batch dimension when feed into tf placeholder
        state = state[np.newaxis, :]

        if np.random.uniform() < epsilon:
            # forward feed the observation and get q value for every actions
            actions_value =session.run(q_values, feed_dict={state_in: state})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, ACTION_DIM)
        return action
# Main learning loop
for episode in range(EPISODE):
    
    # initialize task
    state = env.reset()
    # state = state[np.newaxis,:]

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        # env.render()
        
    # action = explore(state, epsilon)
        action = choose_action(state,epsilon)
        s_, r, done,_ = env.step(action)
        #(np.argmax(action))
        if done:
            r = -1
        
        # store_transition(state,np.argmax(action),reward,next_state)
        store_transition(state,action,r,s_)

        if (step > 3) and (step % 5 == 0):
            if learn_step_counter % replace_target_iter ==0:
                session.run(target_replace_op)
                # print("\n target_params_replaced\n")
            if memory_counter > memory_size:
                sample_index = np.random.choice(memory_size,size=batch_size)
            else:
                sample_index = np.random.choice(memory_counter,size = batch_size)
            batch_memory = memory[sample_index, :]
            # print(batch_memory[:, -STATE_DIM:])

            # q_next, q_eval = session.run([q_next,q_eval],feed_dict={next_state: batch_memory[:,-STATE_DIM:], state: batch_memory[:,:STATE_DIM]})
        
            # q_target = q_eval.copy()
            # batch_index = np.arange(batch_size,dtype = np.int32)
            # eval_act_index = batch_memory[:,STATE_DIM].astype(int)
            # reward = batch_memory[:,STATE_DIM +1]
            # q_target[batch_index,eval_act_index] = reward + GAMMA * np.max(q_next,axis=1)
            # learn_step_counter+=1        

            # nextstate_q_values = q_values.eval(feed_dict={
            #     state_in: [next_state]
            # })

            # TODO: Calculate the target q-value.
            # hint1: Bellman
            # hint2: consider if the episode has terminated
            # target = q_target

            # Do one training step
            # session.run([optimizer], feed_dict={
            #     target_in: [target],
            #     action_in: [action],
            #     state_in: [state]
            # })
            _, cost = session.run(
            [train_op, loss],
            feed_dict={
                # state_in: batch_memory[:, :STATE_DIM],
                # action_in: batch_memory[:, STATE_DIM],
                # reward: batch_memory[:, STATE_DIM + 1],
                # next_state: batch_memory[:, -STATE_DIM:],
                state_in: batch_memory[:, :STATE_DIM],
                action_in: batch_memory[:, STATE_DIM],
                reward: batch_memory[:, STATE_DIM + 1],
                next_state: batch_memory[:, -STATE_DIM:],
            })
            # epsilon = epsilon + epsilon_increment if epsilon < epsilon_max else epsilon_max
            learn_step_counter+=1

        # Update
        state = s_
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            
            for j in range(STEP):
                env.render()
                state = state[np.newaxis, :]
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: state
                    
                }))
                state, r, done, _ = env.step(action)
                total_reward += r
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
