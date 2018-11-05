import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque


# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 100000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.95 # discount factor
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 90  # decay period
EXPLORATION_DECAY = 0.95

REPLAY_SIZE = 10000   #10000
time_step = 0
BATCH_SIZE = 64
LR = 0.0003
time_step = 0

memory_counter = 0
learn_step_counter = 0
replace_target_iter = 30


# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n


# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

def weight_variable(shape):
    initial = tf.random_normal_initializer()
    # initial=tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01,shape=shape)
    return tf.Variable(initial)

# TODO: Define Network Graph
replay_buffer =deque()
w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
# e1 = tf.layers.dense(state_in, 24, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e1')
# q_values = tf.layers.dense(e1, ACTION_DIM, tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer, name='q')
# with tf.variable_scope('eval_net'):
e1 = tf.layers.dense(state_in, 24, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e1')
e2 = tf.layers.dense(e1, 48, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e2')
q_values = tf.layers.dense(e2, ACTION_DIM, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='q')
# v = tf.layers.dense(A, 1, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='ssd')
# q_values = v + (A - (tf.reduce_mean(A,axis=1,keep_dims=True)))
# with tf.variable_scope('target_net'):
#     t1 = tf.layers.dense(NEXT_, 20, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='t1')
#     q_next = tf.layers.dense(t1, ACTION_DIM, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='t2')

# t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
# e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
# with tf.variable_scope('soft_replacement'):
#     target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

# q_action = tf.reduce_sum(tf.multiply(q_values, action_in),reduction_indices =1)


# w1 = weight_variable([STATE_DIM,50])
# b1 = bias_variable([50])
# w2 = weight_variable([50,ACTION_DIM])
# b2 = bias_variable([ACTION_DIM])
# # hidden layer 
# h_layer = tf.nn.relu(tf.matmul(state_in,w1)+b1)
# q_values = tf.add(tf.matmul(layer3,w4), b4, name="predictedReward")

# TODO: Network outputs
# q_values = tf.matmul(h_layer,w2)+b2
q_action = tf.reduce_sum(tf.multiply(q_values, action_in),reduction_indices =1)


# # TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square((target_in - q_action)))
# loss = tf.reduce_mean(temp * target_in)
optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())





 
 
    # q_values_batch = 

# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })[0]
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):
    # print(episode)
    # initialize task
    state = env.reset()
    
    epsilon -=epsilon / EPSILON_DECAY_STEPS
    if(epsilon < FINAL_EPSILON):
        epsilon = FINAL_EPSILON


    # if episode > 50 and epsilon > FINAL_EPSILON:
    #     epsilon -= (INITIAL_EPSILON -FINAL_EPSILON) / 100

    # Update epsilon once per episode
    # epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY_STEPS



    # Move through env according to e-greedy policy
    for step in range(STEP):
        # print(step)
        # env.render()

        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        if done:
            reward = -reward
        replay_buffer.append((state,action,reward,next_state,done))
        # if learn_step_counter % replace_target_iter ==0:
        #     session.run(target_replace_op)
        if len(replay_buffer)> REPLAY_SIZE:
            replay_buffer.popleft()
        if len(replay_buffer)> BATCH_SIZE:
            # print(len(replay_buffer))
            minibatch = random.sample(replay_buffer,BATCH_SIZE)
            # state_batch = [data[0] for data in minibatch]
            # action_batch = [data[1] for data in minibatch]
            # reward_batch = [data[2] for data in minibatch]
            next_state_batch = [var[3] for var in minibatch] 

            target_in_batch=[]
            nextstate_q_values = q_values.eval(feed_dict={
                state_in: next_state_batch
            })
            for i in range(BATCH_SIZE):
                STATE, ACTION, REWARD, NEXT_STATE,TERMINALSATE= minibatch[i]
                predicted_reward = np.max(nextstate_q_values[i])
                if TERMINALSATE:
                    delayerdreward = REWARD
                else:
                    delayerdreward = REWARD + GAMMA*predicted_reward
                target_in_batch.append((STATE,ACTION,delayerdreward))
            states = [var[0] for var in target_in_batch]
            actions = [var[1] for var in target_in_batch]
            rewards = [var[2] for var in target_in_batch]
            # s_ = [var[3] for var in target_in_batch]
            # Do one training step
            session.run([optimizer], feed_dict={
                target_in: rewards,
                action_in: actions,
                state_in: states,
                # NEXT_: s_
            })
            # learn_step_counter +=1
                  
        # if done:
        #     state=env.reset()
        state = next_state
        if done:
            break     


            # TODO: Calculate the target q-value.
            # hint1: Bellman
            # hint2: consider if the episode has terminated
 


        # Update
        # epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY_STEPS
        # epsilon = max(epsilon,FINAL_EPSILON)
        
        
        # if done:
           
        #     break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
