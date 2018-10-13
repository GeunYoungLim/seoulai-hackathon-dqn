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
from seoulai_gym.envs.checkers.base import Constants
from seoulai_gym.envs.checkers.rules import Rules
from seoulai_gym.envs.checkers.utils import board_list2numpy
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

        self.board_enc = BoardEncoding()
        self.board_enc.dark = 1
        self.board_enc.light = -1

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
        model.add(Dense(16, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


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
                for action in actions[i]:
                    target[i][0][0][action] = rewards[i]
            else:
                for classes, action in enumerate(actions[i]):
                    target_action = self.get_action_index(target_val[i][0][0])
                    target[i][0][0][action] = rewards[i] + self.discount_factor * (np.amax(target_action[classes]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def act(self, state):
        board_numpy = board_list2numpy(state, self.board_enc)
        board_numpy = np.reshape(board_numpy, (-1, 8, 8, 1))

        if np.random.rand() <= self.epsilon:
            valid_moves = Rules.generate_valid_moves(state, self.ptype, len(state))
            rand_from_row, rand_from_col = random.choice(list(valid_moves.keys()))
            rand_to_row, rand_to_col = random.choice(valid_moves[(rand_from_row, rand_from_col)])
            action = (rand_from_row, rand_from_col, rand_to_row, rand_to_col)
        else:
            pred = self.model.predict(state)[0]
            action = self.get_action_index(pred)
        return action[0], action[1], action[2], action[3]

    def consume(self, state, action, next_state, reward: float, done: bool):
        state = board_list2numpy(state, self.board_enc)
        state = np.reshape(state, (-1, 8, 8, 1))

        next_state = board_list2numpy(next_state, self.board_enc)
        next_state = np.reshape(next_state, (-1, 8, 8, 1))

        self.append_sample(state, action, reward, next_state, done)
        
        if len(self.memory) >= self.train_start:
            self.train_model()

    def get_action_index(self, pred):
        from_row = pred[0:8]
        from_col = pred[8:16]
        to_row = pred[16:24]
        to_col = pred[24:32]
        action = (np.amax(from_row), np.amax(from_col), np.amax(to_row), np.amax(to_col))
        return action
        
        
