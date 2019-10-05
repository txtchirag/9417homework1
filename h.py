import csv
import numpy as np
import matplotlib.pyplot as plt




def normalize(x,min,max):
    new=((x-min)/(max-min))
    return new

with open('Advertising.csv', 'r') as f:
    table = list(csv.reader(f, delimiter=','))
    data =np.array(table[1:],dtype=np.float)
    print(table[0])

    #1. Pre-­‐processing:

    minValues = np.amin(data, axis=0)[1:4]
    maxValues = np.amax(data, axis=0)[1:4]

    min_TV, min_Radio, min_News = minValues
    max_TV, max_Radio, max_News = maxValues

    #min-max normalization
    norm_data=np.copy(data)
    for i in range(200):
        norm_data[i][1]=normalize(data[i][1],min_TV,max_TV)
        norm_data[i][2] = normalize(data[i][2], min_Radio, max_Radio)
        norm_data[i][3] = normalize(data[i][3], min_News, max_News)


    #2. Creating test and training set
    train_data=np.copy(norm_data[:-10])
    test_data=np.copy(norm_data[-10:])

    #3. Gradient descent
    theta0=-1
    theta1=-0.5
    a=0.01
    max_i=500
