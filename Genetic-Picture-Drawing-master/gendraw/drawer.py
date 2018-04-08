from datetime import datetime
from .circle import Circle
from copy import deepcopy
from PIL import Image
import random
import numpy
import os


__author__ = 'Ilya'


class GeneticDrawer:
    """ Class that draws image based on circles """
    def __init__(self, pattern_picture, n_circles=100, n_generations=100, path=None):
        self.pattern_picture = pattern_picture
        self.n_generations = n_generations
        self.pattern_array = numpy.asarray(self.pattern_picture, dtype='int32')
        self.n_circles = n_circles
        self.mode = self.pattern_picture.mode
        self.size = self.pattern_picture.size
        self.circles = [Circle(self.size, self.mode) for _ in range(self.n_circles)]
        self.image = None
        self.path = path

    def mutate_circles(self):
        # choose circle that would be mutated
        new_circles = deepcopy(self.circles)
        rand_index = random.randint(0, self.n_circles - 1)
        new_circles[rand_index] = Circle(self.size, self.mode)
        return new_circles

    def eval_fitness(self, image):
        current_array = numpy.asarray(image, dtype='int32')
        fitness = abs(self.pattern_array - current_array).sum()
        return fitness

    def show(self):
        if not self.image:
            raise NotImplementedError
        self.image.show()

    def draw_image(self, circles):
        # draw image
        image = Image.new(self.mode, self.size, color='white')
        for circle in circles:
            circle.draw(image)
        # evaluate fitness
        image.fitness = self.eval_fitness(image)
        return image

    def draw(self):
        start = datetime.now()
        self.image = self.draw_image(self.circles)
        for gen in range(int(self.n_generations)):
            mutated_circles = self.mutate_circles()
            offspring = self.draw_image(mutated_circles)
            if offspring.fitness < self.image.fitness:
                self.circles = mutated_circles
                self.image = offspring
            if not gen % 1000 == 0:
                continue
            # save results
            time_passed = ((datetime.now() - start).total_seconds())/60
            print('Generation: {}; Fitness: {}; Time passed: {}'.format(
                gen, self.image.fitness, time_passed))

            if self.path:
                filename = os.path.join(self.path, 'gen_{}.png'.format(gen))
                self.image.save(filename)

