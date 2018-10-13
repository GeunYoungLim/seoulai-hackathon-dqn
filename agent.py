import sys
import pylab
import random
import numpy as np
from collections import deque
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from typing import List
from typing import Tuple

from seoulai_gym.envs.checkers.agents import Agent
from seoulai_gym.envs.checkers.base import Constants
from seoulai_gym.envs.checkers.rules import Rules
from seoulai_gym.envs.checkers.utils import board_list2numpy, generate_random_move
from seoulai_gym.envs.checkers.utils import BoardEncoding


class DQNChecker(Agent):
    def __init__(self, name: str, ptype: int):
        
        if ptype == Constants().DARK:
            name = "RandomAgentDark"
        elif ptype == Constants().LIGHT:
            name = "RandomAgentLight"
        else:
            raise ValueError

        super().__init__(name, ptype)

        self.render = False
        self.load_model = False
 
        # 상태와 행동의 크기 정의
        #self.state_size = (8, 8, 1)
        self.action_size = 4

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_dqn_trained.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        # 8 x 8 -> 4 x 4
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', input_shape=(8, 8, 1)))
        # 4 x 4 -> 2 x 2
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2),activation='relu', padding='same'))
        # 2 X 2 -> 1 x 1
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2),activation='relu', padding='same'))
    
        model.add(Dense(8, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(8, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, 8, 8, 1))
        next_states = np.zeros((self.batch_size, 8, 8, 1))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def act(self, state):
        enc = BoardEncoding()
        enc.dark = 1
        enc.light = -1
        board_numpy = board_list2numpy(state, enc)
        board_numpy = np.reshape(board_numpy, (-1, 8, 8, 1))

        if np.random.rand() <= self.epsilon:
            action = generate_random_move(state, self.ptype, len(state))
        else:
            action = self.model.predict(state)[0]
            #action = np.argmax(q_value[0])
        
        return action[0], action[1], action[2], action[3]

    def consume(self, state, action, next_state, reward: float, done: bool):
        self.append_sample(state, action, reward, next_state, done)
        
        if len(self.memory) >= self.train_start:
            self.train_model()
        
        
