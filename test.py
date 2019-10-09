import numpy as np
import csv

def normalize(x, min, max):
    new = ((x - min) / (max - min))
    return new


with open('Advertising.csv', 'r') as f:
    table = list(csv.reader(f, delimiter=','))
    data = np.array(table[1:], dtype=np.float)

    # 1. Pre-­‐processing:

    minValues = np.amin(data, axis=0)[1:5]
    maxValues = np.amax(data, axis=0)[1:5]

    min_TV, min_Radio, min_News, min_Sales = minValues
    max_TV, max_Radio, max_News, max_Sales = maxValues

    # min-max normalization
    norm_data = np.copy(data)
    for i in range(200):
        norm_data[i][1] = normalize(data[i][1], min_TV, max_TV)
        norm_data[i][2] = normalize(data[i][2], min_Radio, max_Radio)
        norm_data[i][3] = normalize(data[i][3], min_News, max_News)
        norm_data[i][4] = normalize(data[i][4], min_Sales, max_Sales)

    # 2. Creating test and training set
    train_data = np.copy(norm_data[:-10])
    test_data = np.copy(norm_data[-10:])

X = train_data[:,[1]]
y = train_data[:,[4]]

X = np.append(np.ones((len(X),1)),X,axis=1) #append 1's column'

get_theta = lambda theta: np.array([[0, theta]])

thetas = list(map(get_theta, [-1.0, -0.5]))



def cost(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

for i in range(len(thetas)):
    print(cost(X, y, thetas[i]))