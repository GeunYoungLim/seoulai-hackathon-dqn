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

EPISODES = 3000

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
    agent_tag = {}
    agent_tag[a1] = 'Agent_1'
    agent_tag[a2] = 'Agent_2'

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
            from_row, from_col, to_row, to_col = current_agent.act(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(current_agent, from_row, from_col, to_row, to_col)
            action = (from_row, from_col, to_row, to_col)
            #next_state = np.reshape(next_state, [1, state_size])
            
            # # 에피소드가 중간에 끝나면 -100 보상
            if 'invalid_move' in info:
                reward = -10

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
                next_agent.update_target_model()
                # score = score if score == 500 else score + 100
                # 에피소드마다 학습 결과 출력
                for agent in [current_agent, next_agent]:
                    history[agent]['scores'].append(score)
                    history[agent]['episodes'].append(e)
                    tag = agent_tag[agent]

                    pylab.plot(history[agent]['episodes'], history[agent]['scores'], 'b')
                    pylab.savefig("./save_graph/checker_dqn"+ tag +".png")
                    agent.model.save_weights("./save_model/checker_dqn"+ tag + ".h5")

                print('Game over!', current_agent, "agent wins!", "episode:", e, "  score:", score, "  memory length:",
                        len(current_agent.memory), "  epsilon:", current_agent.epsilon)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                
                #sys.exit()
