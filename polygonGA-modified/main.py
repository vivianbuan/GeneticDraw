import csv
import os
import time
from copy import deepcopy
from os import getcwd

#from predict import preload
from MNISTTester import preload
from PIL import Image
from helper_classes import \
    Population, \
    initialize_global_vars, \
    initialize_genetic_vars, \
    set_variables, \
    Genotype, Pixeltype

IMAGE_PATH = os.path.join(os.path.join(getcwd(), 'imgs'), 'digit5.png')
SAVE_PATH = os.path.join(getcwd(), 'save')  # 'img_generate_heroku'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
SAVE_FREQUENCY = 10
STOP_IDX = 500


def scm():
    current_population = Population()
    current_population.generate_initial()

    generation_index = 0
    last_change = 0
    with open(os.path.join(SAVE_PATH, 'fitness_scm.txt'), 'w') as f:
        f.write('')
    with open(os.path.join(SAVE_PATH, 'fitness_scm.csv'), 'w', newline='') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(['generation_index', 'time', 'best_fitness'])
    time_start = time.time()
    while True:

        if generation_index > STOP_IDX:
            break
        current_population.perturbation()
        new_population = deepcopy(current_population)
        new_population.reproduce()
        #new_population.crossover(new_population.select_parents())
        new_population.mutate()
        new_population.elitism()

        if new_population.get_best_fitness() <= current_population.get_best_fitness():
            current_population = new_population
            last_change = generation_index

        if generation_index % SAVE_FREQUENCY == 0:
            print('Saving image "generation_{0:05d}.png"'.format(generation_index))
            current_image = current_population.genotypes[current_population.get_best()].get_image()
            current_image.save(SAVE_PATH + '/generation_{0:05d}.png'.format(generation_index))

            fitness_list = [round(g.get_fitness(), 3) for g in current_population.genotypes]
            log_message = 'generation: {0:05d}, time: {3:}, fitness: {1:}\ngenerations since last update {2:}\n\n' \
                .format(generation_index, str(fitness_list), generation_index - last_change, time.time() - time_start)
            print(log_message)
            with open(os.path.join(SAVE_PATH, 'fitness_scm.txt'), 'a') as f:
                    f.write(log_message)
            info = [generation_index, time.time() - time_start, current_population.get_best_fitness()]
            with open(os.path.join(SAVE_PATH, 'fitness_scm.csv'), 'a', newline='') as my_file:
                wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
                wr.writerow(info)

        generation_index += 1


def simple_ga():
    parent = Genotype()
    parent.generate()

    generation_index = 0
    last_change = 0
    with open(os.path.join(SAVE_PATH, 'fitness_ga.csv'), 'a', newline='') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(['generation_index', 'time', 'best_fitness'])
    time_start = time.time()
    while True:
        if generation_index > STOP_IDX:
            break
        child = deepcopy(parent)
        child.mutate()
        if child.get_fitness() < parent.get_fitness():
            parent = child
            last_change = generation_index

        if not generation_index % SAVE_FREQUENCY:

            print('Saving image "generation_{0:05d}.png"'.format(generation_index))
            current_image = parent.get_image()
            current_image.save(SAVE_PATH + '/generation_{0:05d}.png'.format(generation_index))

            log_message = 'Generation: {0:05d}\nFitness: {1:}\nGenerations since last change: {2:}\n\n' \
                .format(generation_index, round(parent.get_fitness(), 3), (generation_index - last_change))

            print(log_message)
            with open(os.path.join(SAVE_PATH, 'fitness_ga.txt'), 'a') as f:
                f.write(log_message)

            info = [generation_index, time.time() - time_start, parent.get_fitness()]
            with open(os.path.join(SAVE_PATH, 'fitness_ga.csv'), 'a', newline='') as my_file:
                wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
                wr.writerow(info)

        generation_index += 1


def random_pixel():
    parent = Pixeltype()
    parent.generate()

    generation_index = 0
    last_change = 0
    with open(os.path.join(SAVE_PATH, 'fitness_rp.csv'), 'a', newline='') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(['generation_index', 'time', 'best_fitness'])
    time_start = time.time()
    while True:
        if generation_index > STOP_IDX:
            break
        child = deepcopy(parent)
        child.mutate()
        if child.get_fitness() < parent.get_fitness():
            parent = child
            last_change = generation_index

        if not generation_index % SAVE_FREQUENCY:

            print('Saving image "generation_{0:05d}.png"'.format(generation_index))
            current_image = parent.get_image()
            current_image.save(SAVE_PATH + '/generation_{0:05d}.png'.format(generation_index))

            log_message = 'Generation: {0:05d}\nFitness: {1:}\nGenerations since last change: {2:}\n\n' \
                .format(generation_index, round(parent.get_fitness(), 3), (generation_index - last_change))

            print(log_message)
            with open(os.path.join(SAVE_PATH, 'fitness_rp.txt'), 'a') as f:
                f.write(log_message)

            info = [generation_index, time.time() - time_start, parent.get_fitness()]
            with open(os.path.join(SAVE_PATH, 'fitness_rp.csv'), 'a', newline='') as my_file:
                wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
                wr.writerow(info)

        generation_index += 1


def draw_with_reference(algorithm='scm'):
    image = Image.open(IMAGE_PATH)
    initialize_global_vars(image)
    set_variables(IMAGE_MODE='L')
    if algorithm == 'scm':
        scm()
    elif algorithm == 'ga':
        simple_ga()
    else:
        random_pixel()


def draw_ga(algorithm='scm'):
    #initialize_genetic_vars(image_size=(80, 80), img_label=937, img_mode='RGB')
    initialize_genetic_vars(image_size=(56, 56), img_label=0)
    preload()
    if algorithm == 'scm':
        scm()
    elif algorithm == 'ga':
        simple_ga()
    else:
        random_pixel()


if __name__ == '__main__':
    #draw_with_reference(algorithm='scm')
    draw_ga(algorithm='scm')
