import os
import neat
import numpy as np
import tensorflow as tf
from data.data_loader import load_dataset
from dnn.model import setup_hard_model

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

train_ds, val_ds = load_dataset()


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        '''for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2'''
        for example in train_ds:
            image = example['image']
            label = example['label']
            print(image.shape, label)


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-27')
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 300)  # 300 generations

    print('\nBest genome:\n{!s}'.format(winner))

    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    hard_model = setup_hard_model()
    train_images = np.array([images.numpy() for images, _ in train_ds])
    train_labels = np.array([labels.numpy() for _, labels in train_ds])
    print(train_images.shape)
    print(train_labels.shape)
    print(train_images.shape)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)

