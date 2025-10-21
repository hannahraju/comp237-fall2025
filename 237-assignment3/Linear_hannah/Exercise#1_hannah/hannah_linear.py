'''

NAME: Hannah Raju
ID: 301543568
DATE: October 12, 2025
INFO: COMP 237 Assignment 3 - Linear Regression - EXERCISE 1

'''

import numpy as np
import matplotlib.pyplot as plt


# set seed to last two digits of my student number 
np.random.seed(68)

# generate a list x of 100 uniformly distributed random numbers
x = np.random.uniform(size=100)

#generate y = 12x - 4
y = 12*x - 4

# generate scatterplot of x and y
plt.figure()
plt.scatter(x,y, alpha=0.5)
plt.title("Scatterplot of y=12x-4")
plt.xlabel("x (uniformly distributed)")
plt.ylabel("y")
plt.savefig("plot.png")

# generate gaussian noise
noise = np.random.normal(size=100)

#add gaussian noise to the function
y_noise = 12*x - 4 + noise

plt.figure()
plt.scatter(x, y_noise, alpha=0.5)
plt.title("Scatterplot of y = 12 x - 4 + Gaussian noise")
plt.xlabel("x (uniformly distributed)")
plt.ylabel("y")
plt.savefig("plot_noise.png")


