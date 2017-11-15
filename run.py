from network import Network
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Reads in and concatenates data from csv files. Returns data and labels as array
def readData(filenames):
    data_partitions = []
    for filename in filenames:
        data_partitions.append(pd.read_csv(filename))
    csv_data = pd.concat(data_partitions)   #merge files into single datastructure
    csv_data = csv_data.dropna()    #remove entries that contain NaN

    data = csv_data.loc[:,'TRACK_POSITION':]
    data = data.as_matrix()

    labels = csv_data.loc[:, :'SPEED']
    labels = labels.as_matrix()

    return data, labels


# Plot the loss function for different learning rates
def plotLearningRates(data, labels, learning_rates):
    color_range = ['red', 'black', 'blue', 'brown', 'green']
    colors = color_range[:len(learning_rates)]
    for i, color in enumerate(colors): #for every learning rate, train model
        network = Network(data,labels,learning_rates[i])
        network.train()
        plt.loglog(network.loss_saved, color = color, label='${i}$'.format(i=learning_rates[i]))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    filenames = ['aalborg.csv', 'alpine-1.csv', 'f-speedway.csv']
    data, labels = readData(filenames)
    learning_rates = [1e-4, 1e-5, 1e-6]
    plotLearningRates(data, labels, learning_rates)
