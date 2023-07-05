#可改:memory size (無用)
#可改:traning方式, 如新經驗立即訓練 , priority
#可改:reward function
#可改:learning rate

#DDQN
#learning rate = 0.01
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from ENV import NAS
import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import time

np.random.seed(0)
       
class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .97
        self.learning_rate = 0.01
        self.memory = deque(maxlen=1000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
        
    def update_target_model(self):
        #copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    # def immediate_train(self, state, next_state, reward, action, done):
        # a = self.model.predict(state)
        # predict_value = a[0][action]
        # b = self.model.predict(next_state)
        # target = self.gamma * np.amax(b[0]) * (1 - done) + reward
        # error = predict_value - target
        # b = self.target_model.predict(next_state)
        # target = self.gamma * np.amax(b[0]) * (1 - done) + reward
        # target_full = a
        # target_full[0][action] = target
        # self.model.fit(state, target_full, epochs=1, verbose=0)
        # return error
        
    def calculate_error(self, state, next_state, reward, action, done):
        a = self.model.predict(state)
        predict_value = a[0][action]
        b = self.model.predict(next_state)
        target = self.gamma * np.amax(b[0]) * (1 - done) + reward
        return predict_value - target
        
    def act(self, state):           
        act_values = self.model.predict(state)
        print(act_values)
        f = open('Result/result.txt', 'a')
        f.write("-------------------------------\nact_values:{} \n".format(act_values))
        f.close()
        if np.random.rand() <= self.epsilon:
            print('select randomly')
            act_random = random.randrange(self.action_space)
            print('action:',act_random)
            return act_random  
        print('select max')
        act_max = np.argmax(act_values[0])
        print('action:',act_max)
        return act_max

    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        #targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets = rewards + self.gamma*(np.amax(self.target_model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(episode):
    loss = []
    reward_list=[]
    structure_num = 4
    action_space = (structure_num *2) +1
    state_space = (structure_num *2)
    steps = 10
    env = NAS()
    agent = DQN(action_space, state_space)
    count = 0
    iteration_count = 0
    for e in range(episode+1):
        
        last_accuracy = 0
        score = 0
        if e ==0:
            print('-------------Baseline Model------------------')
            reward, next_state, done, accuracy = env.step(0, e , 0, episode)
        else:
            state = next_state
            state = np.reshape(state, (1, state_space))
            for i in range(steps):
                    count += 1
                    print('-------------------------------\nep = ' , e , ' step = ' , i + 1, '\n')
                    action = agent.act(state)
                    reward, next_state, done, accuracy = env.step(action, e , i + 1, episode)

                    iteration_count += 1
                    # action = agent.act(state, env)
                    # if i == steps - 1:
                    # 	reward, next_state, done, accuracy = env.step(action, e + 1, i + 1, episode)
                        
                    # else:
                    #     reward, next_state, done, accuracy = env.step(action, e + 1, i + 1, episode)
                        
                    #print("reward: {} ".format(reward))
                    score = reward
                    next_state = np.reshape(next_state, (1, state_space))
                    # if uniformity < 0:
                    #     last_uniformity = 0
                    # else:
                    last_accuracy= accuracy
                    loss.append(last_accuracy)
                    reward_list.append(score)
                    
                    
                    
                    error = agent.calculate_error(state, next_state, reward, action, done)
                    #error = agent.immediate_train(state, next_state, reward, action, done)
                    print('error = ' + str(error))
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    agent.replay()
                    if done:
                        print("\nepisode: {}/{}, last_accuracy: {}".format(e  , episode, last_accuracy))
                        env.reset()
                        break
                    if i == steps - 1:
                        print("\nepisode: {}/{}, last_accuracy: {}".format(e , episode, last_accuracy))
                        break
            
            
            agent.update_target_model()
    return reward_list, loss, iteration_count

if __name__ == '__main__':

    ep = 100
    
    f = open('Result/result.txt', 'a')
    f.write("Loss History for NAS\n")
    f.close() 
    reward_list, loss, iteration_count = train_dqn(ep)
    plt.plot([i for i in range(iteration_count)], loss)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig('./figure1.png')
    plt.show()
    plt.plot([i for i in range(iteration_count)], reward_list)
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.savefig('./figure2.png')
    plt.show()

    # plt.plot([i for i in range(ep)], loss2)
    # plt.xlabel('episodes')
    # plt.ylabel('score')
    # plt.savefig('./figure2.png')
    # plt.show()
