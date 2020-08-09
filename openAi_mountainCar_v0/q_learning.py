import gym
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()#return initial state

gamma = 0.95 #can change it, same as in assignment(easy21)
learning_rate = 0.1
N0 = 100
alpha = 0.1

SIZE = 30

#this breaks the continuous position and velocity into SIZExSIZE chunks
discrete_observation_space_size = [SIZE] * len(env.observation_space.high)

#this calculates the chunks size: eg: (HI vel - LO vel) / (num. chunks)
discrete_window = (env.observation_space.high - env.observation_space.low) / discrete_observation_space_size


#need to transfor to discrete intervals the velocity and position (state)
def transform_to_discrete(state):
	discrete_state = (state - env.observation_space.low) / discrete_window
	return tuple(discrete_state.astype(np.int))#tuple for the 3 q_values for the 3 possible actions


#explore/exploit
def epsilon_greedy(q, discrete_state, e):
	if (random.random() < e):
		action = random.randint(0, env.action_space.n - 1)
		#print("                                            EXPLORE")
	else:
		action = np.argmax(q[discrete_state])
		#print("-EXPLOIT")
	return action


def q_learning(q, counter, render, episodes, ep_rewards, episode_reward):
		
		discrete_state = transform_to_discrete(env.reset())
		done = False

		while done != True:
			action = None

			if render:
				env.render()

			e = N0 / (N0 + (np.sum(counter[discrete_state + (action, )])))
			action = epsilon_greedy(q, discrete_state, e)
			counter[discrete_state + (action, )] = counter[discrete_state + (action, )] + 1

			
			new_state, reward, done, _ = env.step(action)
			episode_reward = episode_reward + reward

			new_discrete_state = transform_to_discrete(new_state)

			
			#formula from page131 book (2nd ed.)
			if not done:
				#alpha = 1 / (np.sum(counter[discrete_state + (action, )]))
				#print(alpha)
				q_max = np.max(q[new_discrete_state])
				q_curr = q[discrete_state + (action, )]
				q_new = q_curr + alpha * (reward + gamma * q_max - q_curr)
				q[discrete_state + (action, )] = q_new
				

			#if car reached the flag (at position 0.5 in the MountainCar-v0) --> reward = 0
			elif new_state[0] >= env.goal_position:
				q[discrete_state + (action,)] = 0

			#S --> S'
			discrete_state = new_discrete_state
		ep_rewards.append(episode_reward)

	
		return q, ep_rewards, episode_reward


