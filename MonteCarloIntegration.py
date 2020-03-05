import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation as FA
import time
import random

x_range = [1, 2]
y_range = [0, 2.5]
area = (y_range[1] - y_range[0]) * (x_range[1] - x_range[0])
y_text_offset = 0.1 * (y_range[1] - y_range[0])
n = 100

framerate = 24
length_seconds = 60

global fig
global ax

global inside
global total

global last_text

def get_y(x):
	return (1 + np.cos(np.exp(x * x)))
            
def update(frame_number):

	global inside
	global total

	global last_text

	x = random.random() * (x_range[1] - x_range[0]) + x_range[0]
	y = random.random() * (y_range[1] - y_range[0]) + y_range[0]

	if (y <= get_y(x)):
		inside += 1
		plt.plot(x, y, "o", color="green")
	else:
		plt.plot(x, y, "o", color="red")

	total += 1

	text = ax.text(x_range[0], y_range[1] + y_text_offset / 2, "Iterations: " + str(total) + " Result: " + str(area * inside / total))

	if last_text is not None:
		last_text.set_visible(False)

	last_text = text

	print(str(total / framerate) + " seconds " + str(total) + " iterations")

 
if __name__ == "__main__":
	global fig
	global ax

	global inside
	global total

	global last_text

	inside = 0
	total = 0

	last_text = None

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title('Monte-Carlo-Integration')
	plt.plot(x_range, [y_range[1], y_range[1]], color="black")
	x = [i / n for i in range(n* x_range[0], n * x_range[1] + 1)]
	y = [get_y(x_tmp) for x_tmp in x]
	plt.plot(x, y)
	ax.set(xlim=(x_range[0], x_range[1]), ylim=(y_range[0], y_range[1] + y_text_offset))
	random.seed(time.time())
	# Construct the animation, using the update function as the animation director.
	Myanimation = FA(fig, update, frames=(length_seconds * framerate))
	# Set up formatting for the movie files
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=framerate, metadata=dict(artist='Me'), bitrate=1800)
	Myanimation.save("monte-carlo-integration.mp4", writer=writer)
	#plt.show()