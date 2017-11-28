"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import neat
import pandas as pd


# 2-input XOR inputs and expected outputs.

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

# Reads in and concatenates data from csv files. Returns data and labels as array
def readData(filenames):
    data_partitions = []
    for filename in filenames:
        data_partitions.append(pd.read_csv(filename))
    csv_data = pd.concat(data_partitions)   #merge files into single datastructure
    csv_data = csv_data.dropna()    #remove entries that contain NaN
    print(len(csv_data))
    csv_data = csv_data.sample(frac=1).reset_index(drop=True)

    data = csv_data.loc[:,'TRACK_POSITION':]
    data = data.as_matrix()

    labels = csv_data.loc[:, :'SPEED']
    labels = labels.as_matrix()

    return data, labels

#filenames = ['torcs-server/torcs-client/aalborg.csv', 'torcs-server/torcs-client/alpine-1.csv', 'torcs-server/torcs-client/f-speedway.csv']
#data, labels = readData(filenames)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(data, labels):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


def do_the_thing(data, labels): 
	
	# Load configuration.
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
		                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
		                 'config-feedforward')

	# Create the population, which is the top-level object for a NEAT run.
	p = neat.Population(config)

	# Add a stdout reporter to show progress in the terminal.
	p.add_reporter(neat.StdOutReporter(False))

	# Run until a solution is found.
	winner = p.run(eval_genomes)

	# Display the winning genome.
	#print('\nBest genome:\n{!s}'.format(winner))

	# Show output of the most fit genome against training data.
	#print('\nOutput:')
	winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
	#for xi, xo in zip(data, labels):
	#	output = winner_net.activate(xi)
	#print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
	return winner_net
