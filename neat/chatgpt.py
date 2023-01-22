import os
import numpy as np
import neat
from PIL import Image


def preprocess_images():
    ellipse = Image.open('../../neat_images/ellipse.png').convert('L')
    square = Image.open('../../neat_images/square.png').convert('L')
    triangle = Image.open('../../neat_images/triangle.png').convert('L')

    ellipse_data = np.array(ellipse).flatten()
    square_data = np.array(square).flatten()
    triangle_data = np.array(triangle).flatten()

    return [triangle_data, square_data, ellipse_data]


def calculate_fitness(output):
    return 1


def evaluate_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # Use preprocessed image data as input to the network
        images = preprocess_images()
        for img in images:
            output = net.activate(img)
            genome.fitness = calculate_fitness(output)


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

    winner = p.run(evaluate_genomes, 300)  # 300 generations


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
