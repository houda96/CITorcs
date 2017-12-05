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

    data = csv_data.loc[:,'SPEED':]
    data = data.as_matrix()

    labels = csv_data.loc[:, :'STEERING']
    max_target_speed = labels["TARGET_SPEED"].max()

    labels["TARGET_SPEED"] = labels["TARGET_SPEED"].apply(lambda x: x/max_target_speed)
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
