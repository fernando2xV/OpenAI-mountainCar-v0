import numpy as np
from q_learning import *
import matplotlib.pyplot as plt
from matplotlib import style

#----------------------video stuff
import cv2
import os
style.use("ggplot")
#----------------------

episodes = 2000000
show_each = 500
save_each = 10

NUMBER = 1197080 #this number is the number of 1 tables I saved during training, its used when making the video (hardcoded this because training stopped)

episode_rewards = []
aggregate_episodes_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

#change this to render or not (True or False)
render = True

#SIZE x SIZE x 3
q = np.zeros((discrete_observation_space_size + [env.action_space.n]))
counter = np.zeros((discrete_observation_space_size + [env.action_space.n]))



for i in range(episodes + 1):
	episode_reward = 0
	q, episode_rewards, episode_reward = q_learning(q, counter, render, i, episode_rewards, episode_reward)

	if i % show_each == 0:
		#render = True
		average_reward = sum(episode_rewards[ - show_each:]) / len(episode_rewards[- show_each:])
		aggregate_episodes_rewards['ep'].append(i)
		aggregate_episodes_rewards['avg'].append(average_reward)
		aggregate_episodes_rewards['min'].append(min(episode_rewards[-show_each:]))
		aggregate_episodes_rewards['max'].append(max(episode_rewards[-show_each:]))
		print(f"Episode: {i} avg: {average_reward} min: {min(episode_rewards[-show_each:])} max: {max(episode_rewards[-show_each:])}")
	else:
		render = False

	#each 10 episodes save the q table
	if i % save_each == 0:
		np.save(f"qtables_car/{i}-q_table.npy", q)

env.close()


plt.plot(aggregate_episodes_rewards['ep'], aggregate_episodes_rewards['avg'], label = "avg")
plt.plot(aggregate_episodes_rewards['ep'], aggregate_episodes_rewards['min'], label = "min")
plt.plot(aggregate_episodes_rewards['ep'], aggregate_episodes_rewards['max'], label = "max")
plt.legend(loc = 4)
plt.show()

'''
#--------------------------Video/Plot stuff-------------------------------

def graph_colours(value, vals):
	if value == max(vals):
		return "green", 1.0
	else:
		return "red", 0.3


fig = plt.figure(figsize = (12, 9))

for i in range(0, episodes, 10):
	print(i)
	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)

	q_tables = np.load(f"qtables_car/{i}-qtable.npy")

	for x, x_vals in enumerate(q_tables):
		for y, y_vals in enumerate(x_vals):
			ax1.scatter(x, y, c = graph_colours(y_vals[0], y_vals)[0], marker = "o", alpha = graph_colours(y_vals[0], y_vals)[1])
			ax2.scatter(x, y, c = graph_colours(y_vals[1], y_vals)[0], marker = "o", alpha = graph_colours(y_vals[1], y_vals)[1])
			ax3.scatter(x, y, c = graph_colours(y_vals[2], y_vals)[0], marker = "o", alpha = graph_colours(y_vals[2], y_vals)[1])

			ax1.set_ylabel("Action 0")
			ax2.set_ylabel("Action 1")
			ax3.set_ylabel("Action 2")
	plt.savefig(f"qtables_charts/{i}.png")
	plt.clf()





def do_video():
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('qlearning.avi', fourcc, 60.0, (1200, 900))

	for i in range(0, NUMBER, 10):
		image_path = f"qtables_charts/{i}.png"
		print(image_path)
		frame = cv2.imread(image_path)
		out.write(frame)

	out.release()

do_video()
'''