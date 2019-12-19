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
import csv

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
CREDIT_INTERVAL_TIME = 0.4
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

def save_results(trace_timesteps, trace_feedback, trace_retro_feedbacks, filename):
    table = [["Timesteps", "Feedbacks", "Retro-Feedbacks"]]
    
    for i in range(len(trace_timesteps)):
        row = []
        row.append(trace_timesteps[i])
        row.append(trace_feedback[i])
        row.append(trace_retro_feedbacks[i])
        table.append(row)
    
    with open(filename, "w+") as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in table]
        
        

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    print('state size:' ,state_size)
    print('action size: ', action_size)
    done = False
    batch_size = 128
    trace_timesteps = []
    trace_feedbacks = []
    trace_retro_feedbacks = []
    scores = 0
    count_seed = 0

    name = input("Please enter your name: ")
    filename = "results/Retro_TAMER_single_" + name + ".csv"
    for e in range(EPISODES):
        legal_input = False
        while (not legal_input):
            user_input = input("Continue training? [y/n]: ")
            if user_input == "n":
                save_results(trace_timesteps, trace_feedbacks, trace_retro_feedbacks, filename)
                legal_input = True
                exit(1)
            elif user_input == "y":
                legal_input = True
                pass
            else:
                print("Illegal input, must be [y/n]")
        
        env.seed(seed=count_seed)
        state = env.reset()
        init_state = copy.deepcopy(state)
        state = np.reshape(state, [1, state_size])
        flag = 0
        h = 0.0
        count = 0
        num_feedbacks = 0
        
        full_history = []
        history = [] # Last is newest, first is oldest
        while True:
            count += 1
            env.render()            
            action = agent.act(env, state)
            print(ACTION_NAMES[action])
            
            next_state, reward, done, info = env.step(action)
            done = bool(env.state[0] >= env.goal_position and env.state[1] >= env.goal_velocity)
                        
            # Note: assigns multiple times
            if (keyboard.is_pressed('left')):
                h = -1.0
            elif (keyboard.is_pressed('right')):
                h = 1.0           

            # give more reward if the cart reaches the flag in 200 steps
            if done:
                print("giving reward")
                print("position " + str(env.state[0])) #position
                print("velocity " + str(env.state[1])) #velocity
                print("goal position " + str(env.goal_position))
                print("goal velocity " + str(env.goal_velocity))
                print(bool(env.state[0] >= env.goal_position and env.state[1] >= env.goal_velocity))
            
            next_state = np.reshape(next_state, [1, state_size])
            
            
            curr_time = time.time()
            full_history.append((state, action, next_state, done, curr_time))
            update_history(history, state, action, next_state, done, curr_time)
            if h != 0:
                time.sleep(STOP_TIME)
                print(str(h) + " credit assigned")
                num_feedbacks += 1
                to_remember = assign_credit(history)
                # get all state, action, reward, next_state, done that we need to train on
                
                # agent.remember on all of them from 0.
                for s, a, next_s, d, act_time in to_remember: # to_remember is list of tuples of data to assign credit to
                    # More nuanced time steps?
                    reward = h / len(to_remember)
                    agent.remember(s, a, reward, next_s, d)
                
                agent.replay(len(to_remember))
                agent.memory = deque(maxlen=2000)
                h = 0.0
            
            state = next_state
            scores += reward
            if done:
                flag = 1
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, scores, agent.epsilon))
                break
            
        trace_timesteps.append(count)
        trace_feedbacks.append(num_feedbacks)
        
        # Simulation
        env.seed(seed=count_seed)
        sim_init_state = env.reset()
#         print(sim_init_state)
#         print(init_state)
#         if (sim_init_state[0] == init_state[0] and sim_init_state[1] == init_state[1]):
#             print("Same initial states:") #yaaas
#             print("sim_init_state:")
#             print(sim_init_state)
#             print("init_state:")
#             print(init_state)
#             print("")
            
# #         full_sim_state_history = []
# #         for a in full_action_history:
# #             print("action")
# #             print(a)
            
# #             full_sim_state_history.append(env.state)
# #             env.step(a)
# #             env.render()
        
        # allow human to move forward in the simulation by pressing the "up" key
        max_actions = len(full_history)
        count_actions = 0
        h = 0.0
        num_remembered = 0
        num_retro_feedbacks = 0
        while True:
            if count_actions >= max_actions:
                break
            if keyboard.is_pressed('up'):
                s, action, next_s, d, act_time  = full_history[count_actions]
                print(ACTION_NAMES[action])
                env.step(action)
                env.render()
                count_actions += 1
            
            if (keyboard.is_pressed('left')):
                h = -1.0
            elif (keyboard.is_pressed('right')):
                h = 1.0
            
            if h != 0.0:
                num_retro_feedbacks += 1
                time.sleep(STOP_TIME)
                print(str(h) + " credit assigned")
                reward = h
                agent.remember(s, action, reward, next_s, d)  
                num_remembered += 1
                h = 0.0
        
        if num_remembered > 0:
            agent.replay(num_remembered)
            agent.memory = deque(maxlen=2000)
            
        trace_retro_feedbacks.append(num_retro_feedbacks) 
            
#         if (len(full_state_history) != len(full_sim_state_history)):
#             print("Mismatching history lengths:")
#             print("Length of State History: " + str(len(full_state_history)))
#             print("Length of Sim State History: " + str(len(full_sim_state_history)) + "\n")
#             exit(1)
            
#         for i in range(len(full_state_history)):
#             print("Action taken: " + str(full_action_history[i]))
#             print("State History at " + str(i))
#             print(full_state_history[i])
#             print("Sim State History at " + str(i))
#             print(full_sim_state_history[i])

        if flag == 0:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, t, agent.epsilon))      
#         if e % 100 == 0:
#             print('saving the model')
#             agent.save("mountain_car-dqn.h5")
        time.sleep(2)
        count_seed += 1
