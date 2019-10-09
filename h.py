import csv
import numpy as np
import matplotlib.pyplot as plt


def normalize(x, min, max):
    new = ((x - min) / (max - min))
    return new

with open('Advertising.csv', 'r') as f:
    table = list(csv.reader(f, delimiter=','))
    data = np.array(table[1:], dtype=np.float)


# 1. Pre-­‐processing:

minValues = np.amin(data, axis=0)[1:4]
maxValues = np.amax(data, axis=0)[1:4]

min_TV, min_Radio, min_News = minValues
max_TV, max_Radio, max_News = maxValues

# min-max normalization

for i in range(200):
    data[i][1] = normalize(data[i][1], min_TV, max_TV)
    data[i][2] = normalize(data[i][2], min_Radio, max_Radio)
    data[i][3] = normalize(data[i][3], min_News, max_News)


# 2. Creating test and training set
train_data = np.copy(data[:-10])
test_data = np.copy(data[-10:])

# 3. Gradient descent
theta0 = -1
theta1 = -0.5
theta = np.array([[theta0, theta1]])

alpha = 0.01
max_iter = 500

#train TV x Sales
X = train_data[:,[1]]
y = train_data[:,[4]]
m=X.shape[0] #no.of training data items
cost=np.ones(shape=(max_iter,1))
for i in range(max_iter):
    cost[i] = 1 / (2 * m) * sqfunc(X, y, theta)
    theta = np.array([(theta[0][0] + ((alpha / m) * sumfunc0(X,y,theta))).item(),(theta[0][1] + ((alpha / m) * sumfunc1(X,y,theta))).item()])




# plt.scatter(data[:, 4], data[:, 1])
# plt.show()
