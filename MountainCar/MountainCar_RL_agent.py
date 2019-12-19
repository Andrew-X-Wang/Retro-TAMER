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
import sys
import csv

EPISODES = 100


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.05 #1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.002
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
#         # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=2))
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
#         model = SGDRegressor()
#         features = np.zeros(4)
#         model.partial_fit(np.array([features]), np.array([0.0]))
        
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, env, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
#         action_values = []
#         for action in range(self.action_size):
#             features = self.get_rbf_features(env, action)
#             action_values.append(self.model.predict(np.array([features]))[0])
            
# #         print(action_values)
#         action_values = np.array(action_values)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # np.random.choice(np.flatnonzero(action_values == action_values.max()))  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
#         features = []
#         rewards = []
        for state, action, reward, next_state, done in minibatch:
#             features.append(np.array(self.get_rbf_features_with_next_state(state[0], next_state[0])))
#             rewards.append(reward)
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future  = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(Q_future)
                
            self.model.fit(state, target, epochs=1, verbose=0)
        
#         print(features)
#         print(rewards)
#         self.model.partial_fit(np.array(features), np.array(rewards))
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

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
STOP_TIME = 0.1

ACTION_NAMES = ["LEFT", "STILL", "RIGHT"]        
    
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

def save_results(trace_timesteps, trace_feedbacks, trainer_experience, first_trial, filename):
    table = [["Timesteps"]]
    
    for i in range(len(trace_timesteps)):
        row = []
        row.append(trace_timesteps[i])
        table.append(row)
    
    with open(filename, "w+") as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in table]
        
def save_run(seed, final_run_history, filename):   
    table = [["state", "action", "next_state", "done", "curr_time", "seed"]]
    
    for i in range(len(final_run_history)):
        row = []
        for j in range(len(final_run_history[i])):
            row.append(final_run_history[i][j])
        if i == 0:
            row.append(seed)
        table.append(row)
    
    with open(filename, "w+") as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in table]

        

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    print('state size: ', state_size)
    print('action size: ', action_size)
    done = False
    batch_size = 64
    trace_timesteps = []
    scores = 0
    
    for e in range(EPISODES):
        
        state = env.reset()
        init_state = copy.deepcopy(state)
        state = np.reshape(state, [1, state_size])
        flag = 0
        h = 0.0
        count = 0
        
        while True:
            if count % 500 == 0:
                print(count)
            count += 1
#             env.render()
            action = agent.act(env, state)
            next_state, reward, done, info = env.step(action)
            done = bool(env.state[0] >= env.goal_position and env.state[1] >= env.goal_velocity)       

            
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)            
            state = next_state
            scores += reward
            if done:
                flag = 1
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, scores, agent.epsilon))
                break
                
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)
                agent.memory = deque(maxlen=2000)
                
        print("Timesteps: %d" % (count))
        trace_timesteps.append(count)
               

        if flag == 0:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, t, agent.epsilon))      
        if e % 100 == 0:
            print('saving the model')
            agent.save("mountain_car-dqn.h5")

    filename = "results/DQN_normal_state_space.csv"
    save_results(trace_timesteps, filename)
