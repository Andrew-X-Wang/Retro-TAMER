import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.linear_model import SGDRegressor

import code
import keyboard
import time
import copy
import math

EPISODES = 400


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.0 #0.95    # discount rate
        self.epsilon = 0.0 #1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.002
        self.model = self._build_model()
        self.target_model = self._build_model()
#         self.update_target_model()

    def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Dense(64, input_dim=2))
#         model.add(Activation('relu'))

# #         model.add(Dense(64))
# #         model.add(Activation('relu'))

#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss="mean_squared_error",
#                       optimizer=Adam(lr=self.learning_rate))
        model = SGDRegressor()
        features = np.zeros(4)
        model.partial_fit(np.array([features]), np.array([0.0]))
        
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, env, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        action_values = []
        for action in range(self.action_size):
            features = self.get_rbf_features(env, action)
            action_values.append(self.model.predict(np.array([features]))[0])
            
#         print(action_values)
        action_values = np.array(action_values)
        return np.random.choice(np.flatnonzero(action_values == action_values.max())) #np.argmax(action_values)  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        features = []
        rewards = []
        for state, action, reward, next_state, done in minibatch:
            features.append(np.array(self.get_rbf_features_with_next_state(state[0], next_state[0])))
            rewards.append(reward)
#             target = self.model.predict(state)
#             if done:
#                 target[0][action] = reward
#             else:
#                 Q_future  = self.target_model.predict(next_state)[0]
#                 target[0][action] = reward + self.gamma * np.amax(Q_future)
                
#             self.model.fit(state, target, epochs=1, verbose=0)
        
#         print(features)
#         print(rewards)
        self.model.partial_fit(np.array(features), np.array(rewards))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

    def simulate_step(self, env, action):
        assert env.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = env.state
        velocity += (action-1)*env.force + math.cos(3*position)*(-env.gravity)
        velocity = np.clip(velocity, -env.max_speed, env.max_speed)
        position += velocity
        position = np.clip(position, env.min_position, env.max_position)
        if (position==env.min_position and velocity<0): velocity = 0

        done = bool(position >= env.goal_position and velocity >= env.goal_velocity)
        reward = -1.0

    #     self.state = (position, velocity)
        next_state = (position, velocity)
        return np.array(next_state), reward, done, {}

    def get_rbf_features(self, env, action):
        next_state, reward, done, info = self.simulate_step(env, action)
        next_state = [[next_state[0]], [next_state[1]]]
        state = [[env.state[0]], [env.state[1]]]
        return polynomial_kernel(state, next_state, degree=2).flatten()
#         return rbf_kernel(state, next_state).flatten()
    
    def get_rbf_features_with_next_state(self, state, next_state):
        next_state = [[next_state[0]], [next_state[1]]]
        state = [[state[0]], [env.state[1]]]
        return polynomial_kernel(state, next_state, degree=2).flatten()
#         return rbf_kernel(state, next_state).flatten()
        
    
CREDIT_MIN_TIME = 0.2
CREDIT_MAX_TIME = 0.6
        
    
def update_history(history, state, action, next_state, done, curr_time):
    history.append((state, action, next_state, done, curr_time))
    while (curr_time - history[0][4] > CREDIT_MAX_TIME):
        history.pop(0)
        
# Returns list of tuples to be appended directly to memory
def assign_credit(history):
    to_remember = []
    i = 0
    curr_time = history[-1][4]
    while (curr_time - history[i][4] > CREDIT_MIN_TIME):
        to_remember.append(copy.deepcopy(history[i]))
        i += 1
        
    return to_remember


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    history = [] # Last is newest, first is oldest


    agent = DQNAgent(state_size, action_size)
#     ############### TAMER code ################################
#     model = sklearn.linear_model.SGDRegressor()
# #     model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes = 50)
#     features = np.zeros(state_size)
#     h = 0.0
#     model.partial_fit(np.array([features]), np.array([-1.0]))
    
#     memory = {"Boards": [], "Scores": [], "Levels": []}

#     ############### TAMER code ################################

    print('state size:' ,state_size)
    print('action size: ', action_size)
    done = False
    batch_size = 128

    scores = 0
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        flag = 0
        h = 0.0
        count = 0
        while True:
            count += 1
            # uncomment this to see the actual rendering 
            env.render()            
            action = agent.act(env, state)
#             if count % 10 == 0:
            print(action)
            
            next_state, reward, done, info = env.step(action)
            done = bool(env.state[0] >= env.goal_position and env.state[1] >= env.goal_velocity)
            
            # Note: assigns multiple times
            if (keyboard.is_pressed('left')):
                h = -10.0
            elif (keyboard.is_pressed('right')):
                h = 10.0
#             else:
#                 h = 0.0

            if next_state[1] > state[0][1] and next_state[1]>0 and state[0][1]>0:
                reward += 15
            elif next_state[1] < state[0][1] and next_state[1]<=0 and state[0][1]<=0:
                reward +=15
            

            # give more reward if the cart reaches the flag in 200 steps
            if done:
                print("giving reward")
                print("position " + str(env.state[0])) #position
                print("velocity " + str(env.state[1])) #velocity
                print("goal position " + str(env.goal_position))
                print("goal velocity " + str(env.goal_velocity))
                print(bool(env.state[0] >= env.goal_position and env.state[1] >= env.goal_velocity))
#                 reward = reward + 10000
            else:
                # put a penalty if the no of t steps is more
#                 reward = reward - 10  
                pass
            next_state = np.reshape(next_state, [1, state_size])
            
            
            curr_time = time.time()
            update_history(history, state, action, next_state, done, curr_time)
            if h != 0:
                print("credit assigned")
                to_remember = assign_credit(history) #@TODO: not implemented, use uniform distribution from min to max to assign credit
#                 print(len(to_remember))
                # get all state, action, reward, next_state, done that we need to train on
                
                # agent.remember on all of them from 0.
                for state, action, next_state, done, action_time in to_remember: # to_remember is list of tuples of data to assign credit to
                    # More nuanced time steps?
                    reward = h / len(to_remember)
                    agent.remember(state, action, reward, next_state, done)
                
                agent.replay(len(to_remember))
                agent.memory = deque(maxlen=2000)
                h = 0.0
                
            
            
            state = next_state
            scores += reward
            if done:
                flag = 1
#                 agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, scores, agent.epsilon))
                break
            
            # print("Scores: " + str(scores))
            # print("Reward: " + str(reward))


#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)
#                 # Reset memory? Need to keep and queue/dequeue appropriately
#                 agent.memory = deque(maxlen=2000)
               
            
        if flag == 0:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, t, agent.epsilon))      
        if e % 100 == 0:
            print('saving the model')
#             agent.save("mountain_car-dqn.h5")
