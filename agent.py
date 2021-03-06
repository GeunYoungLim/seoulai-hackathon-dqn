import sys
import random
import numpy as np
from collections import deque
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose

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
    def __init__(self, name: str, ptype: int, load_model: bool = True, epsilon: float = 0.0):
        self.board_enc = BoardEncoding()
        if ptype == Constants().DARK:

            self.board_enc.dark = 0.5
            self.board_enc.light = -0.5
            self.board_enc.dark_king = 1.
            self.board_enc.light_king = -1.

        elif ptype == Constants().LIGHT:
            
            self.board_enc.dark = -0.5
            self.board_enc.light = 0.5
            self.board_enc.dark_king = -1.
            self.board_enc.light_king = 1.
        else:
            raise ValueError

        super().__init__(name, ptype)

        self.render = False
        self.load_model = load_model
 
        # 상태와 행동의 크기 정의
        #self.state_size = (8, 8, 1)
        self.action_size = 4

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = epsilon
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.00
        self.batch_size = 64
        self.train_start = 3000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=4000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()
        
        if self.load_model:
            self.model.load_weights("./save_model/checker_dqn.h5")


    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        # 8 x 8 -> 4 x 4
        model.add(Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', input_shape=(8, 8, 1)))
        # 4 x 4 -> 2 x 2
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2),activation='relu', padding='same'))
        # 2 X 2 -> 1 x 1
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2),activation='relu', padding='same'))
        # 1 X 1 -> 2 X 2
        model.add(Conv2DTranspose(16 ,kernel_size=(3, 3), strides=(2, 2), padding='same'))
        # 2 x 2 -> 4 x 4
        model.add(Conv2DTranspose(8 ,kernel_size=(3, 3), strides=(2, 2), padding='same'))
        # 4 X 4 -> 4 X 8
        model.add(Conv2DTranspose(1 ,kernel_size=(3, 3), strides=(1, 2), padding='same'))
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
                for classes, action in enumerate(actions[i]):
                    target[i][classes][action] = rewards[i]
            else:
                for classes, action in enumerate(actions[i]):
                    target_action = self.get_action_index(target_val[i][0])
                    target[i][classes][action] = rewards[i] + self.discount_factor * (np.amax(target_action[classes]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def act(self, state):
        raw_state = state
        state = board_list2numpy(state, self.board_enc)
        state = np.reshape(state, (-1, 8, 8, 1))

        if np.random.rand() <= self.epsilon:
            valid_moves = Rules.generate_valid_moves(raw_state, self.ptype, len(raw_state))
            rand_from_row, rand_from_col = random.choice(list(valid_moves.keys()))
            rand_to_row, rand_to_col = random.choice(valid_moves[(rand_from_row, rand_from_col)])
            action = (rand_from_row, rand_from_col, rand_to_row, rand_to_col)
        else:
            pred = self.model.predict(state)[0]
            action = self.get_action_index(pred)

            p_from_row = int(action[0])
            p_from_col = int(action[1])
            p_to_row = int(action[2])
            p_to_col = int(action[3])

            if not Rules.validate_move(raw_state, p_from_row, p_from_col, p_to_row, p_to_col):
                print('-', end='', flush=True)
                valid_moves = Rules.generate_valid_moves(raw_state, self.ptype, len(raw_state))
                rand_from_row, rand_from_col = random.choice(list(valid_moves.keys()))
                rand_to_row, rand_to_col = random.choice(valid_moves[(rand_from_row, rand_from_col)])
                action = (rand_from_row, rand_from_col, rand_to_row, rand_to_col)
            else:
                print('@', end='', flush=True)

        return int(action[0]), int(action[1]), int(action[2]), int(action[3])
    def consume(self, obs: List[List], reward: float, done: bool, **kwargs):
    #def consume(self, state, action, next_state, reward: float, done: bool):
        if not self.epsilon > 0:
            return
        for key in ['action', 'next_state'] :
            if key not in kwargs:
                print('not train')
                return
        
        state = obs
        action = kwargs['action']
        next_state = kwargs['next_state']

        state = board_list2numpy(state, self.board_enc)
        state = np.reshape(state, (-1, 8, 8, 1))

        next_state = board_list2numpy(next_state, self.board_enc)
        next_state = np.reshape(next_state, (-1, 8, 8, 1))
        
        self.append_sample(state, action, reward, next_state, done)
        
        if len(self.memory) >= self.train_start:
            self.train_model()

    def get_action_index(self, pred):
        from_row = pred[0]
        from_col = pred[1]
        to_row = pred[2]
        to_col = pred[3]
        action = (np.argmax(from_row), np.argmax(from_col), np.argmax(to_row), np.argmax(to_col))
        return action
        
        
