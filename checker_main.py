"""
seoulai hackathon 2018
"""
import pylab
import numpy as np

import seoulai_gym as gym
from seoulai_gym.envs.checkers.agents import RandomAgentLight
from seoulai_gym.envs.checkers.agents import RandomAgentDark
from seoulai_gym.envs.checkers.base import Constants

from agent import DQNChecker

EPISODES = 300

RENDER = True

if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    #env = gym.make('CartPole-v1')
    env = gym.make("Checkers")

    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n

    # DQN 에이전트 생성
    a1 = DQNChecker("Agent 1", Constants().DARK)
    a2 = DQNChecker("Agent 2", Constants().LIGHT)

    history = {}
    history[a1] = {'scores': [], 'episodes': []}
    history[a2] = {'scores': [], 'episodes': []}

    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
#        state = env.reset()
        state = env.reset()
        
        current_agent = a1
        next_agent = a2


        while not done:
            if RENDER:
                env.render()

            # 현재 상태로 행동을 선택
            action = current_agent.act(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            #next_state = np.reshape(next_state, [1, state_size])
            
            # # 에피소드가 중간에 끝나면 -100 보상
            # reward = reward if not done or score == 499 else -100

            current_agent.consume(state, action, next_state, reward, done)

            score += reward
            state = next_state

            # switch agents
            temporary_agent = current_agent
            current_agent = next_agent
            next_agent = temporary_agent

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                current_agent.update_target_model()

                # score = score if score == 500 else score + 100
                # 에피소드마다 학습 결과 출력
                history[current_agent]['scores'].append(score)
                history[current_agent]['episodes'].append(e)

                pylab.plot(history[current_agent]['scores'], history[current_agent]['episodes'], 'b')
                pylab.savefig("./save_graph/agent1/checker_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(current_agent.memory), "  epsilon:", current_agent.epsilon)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                current_agent.model.save_weights("./save_model/checker_dqn.h5")
                #sys.exit()
