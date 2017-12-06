import numpy as np
from network import Network
import helper_functions
from freestyle import do_the_thing
import shlex, subprocess
import os
import torch
import sys

# First make a regular network
# Then make sure that it is riding and keep the amount of distance
# Now make n children from that car with random noise
# Let each of these children race as well and get their amount of distance


# Initialize the "best" network
# filenames = ['aalborg.csv', 'alpine-1.csv', 'f-speedway.csv', 'data_track_2.csv']
filenames = ['aalborg.csv', 'alpine-1.csv', 'f-speedway.csv']
# filenames = ['data_track_2.csv']
data, labels = helper_functions.readData(filenames)
# self.network = do_the_thing(data, labels)
learning_rate = 1e-5
network = Network(data, labels, learning_rate)
network.train()
torch.save(network, "current_network.pt")

# Begin with the first parent car, let him ride and store
max_at_level = 1
amount_children = 5

command = "./start.sh & torcs -r ~/School/Master/FirstYear/CI/Project/torcs-server/sample-config/quickrace.xml"
os.system(command)

# Maybe other manner than the distance_raced?? If the car always rides at least
with open("distance_raced.txt") as file:
    data = file.read()
print(data)
fitness = float(data)
parent_weights = list(network.model.parameters())
population = [(network.model.parameters(), fitness)]

# The amount of generations
for GENERATION in range(2):
    for i in range(amount_children):
        for ind, param in enumerate(network.model.parameters()):
            param.data = parent_weights[ind].data + np.random.normal(0,1)
        # Let the driver race
        torch.save(network, "current_network.pt")
        os.system(command)

        # Get outcome race
        with open("distance_raced.txt") as file:
            data = file.read()
        if data:
            fitness = float(data)
        else:
            fitness = 0.0
        population.append((network.model.parameters(), fitness))

    # Get the fittest racer and keep him for the next generation
    print(population)
    parent_weights, fitness_best = max(population, key=lambda item:item[1])
    population = [list(parent_weights), fitness_best]

# To show the result, let the best car ride for now
for ind, param in enumerate(network.model.parameters()):
    param.data = parent_weights[ind].data + np.random.normal(0,1)
torch.save(network, "current_network.pt")

# Now start torcs in the terminal to see it race



#args = shlex.split(command)
#print(args)
#subprocess.Popen(args, shell=False)
#output = subprocess.run(["./start.sh"], stdout=subprocess.PIPE)
#print(output)
