import math
from copy import deepcopy
from math import sqrt
from random import randrange, uniform, shuffle, random, sample

import adapt_util
import matplotlib.pyplot as plt
import numpy as np
import pylab
#from predict import predict
from MNISTTester import predict
from PIL import Image, ImageDraw
import bisect

POPULATION_SIZE = 10
NUMBER_OF_POLYGONS = 50
MIN_VERTICES = 3
MAX_VERTICES = 5
NUMBER_OF_PARENTS = 4
ELITISM_NUMBER = 4  # Number of fittest genotypes to carry to the next generation directly.
OFFSET = 10
PROBABILITY_MUTATION = 1
POLY_DIAMETER = 0.25
POLYGON_RATE = 0.5
CHALLENGE_SCORE = 1e-3
IMAGE_MODE = 'L'
TRANSPARENCY = False

INPUT_IMAGE = None
IMAGE_WIDTH = 0
IMAGE_HEIGHT = 0
IMAGE_MATRIX = None
MAX_DELTA = 0
CLASSIFIER = None
IMAGE_LABEL = None


class Shape:
    def __init__(self):
        self.color = [0, 0, 0, 255]
        self.mode = IMAGE_MODE
        self.vertices = []
        self.transparency = TRANSPARENCY
        self.coord = []

    def get_coord(self):
        return self.coord

    def __lt__(self, other):
        this_coord = self.get_coord()
        other_coord = other.get_coord()
        if this_coord[0] < other_coord[0]:
            return True
        elif this_coord[0] > other_coord[0]:
            return False
        else:
            return this_coord[1] < other_coord[1]

    def generate(self, vertices=True, color=True):
        pass
        '''
        if color and self.randomize_color:
            self.color = generate_color(self.mode)
        if vertices:
            self.vertices = []  # To handle mutation.
            for i in range(randrange(MIN_VERTICES, MAX_VERTICES + 1)):
                self.vertices.append(generate_point(IMAGE_WIDTH, IMAGE_HEIGHT))
        '''

    def mutate(self):
        self.generate()
        #rand = random() < 0.5
        #self.generate(vertices=rand, color=not rand)  # Mutate either color or vertices.

    def draw(self):
        pass

    def rotated_about(self, ax, ay, bx, by, angle):
        '''
        https://stackoverflow.com/questions/34747946/rotating-a-square-in-pil
        rotates point `A` about point `B` by `angle` radians clockwise.
        '''
        radius = math.sqrt((by - ay) ** 2 + (bx - ax) ** 2)
        angle = math.atan2(ay - by, ax - bx) - angle
        return (
            round(bx + radius * math.cos(angle)),
            round(by + radius * math.sin(angle))
        )

    def rotate(self, angle):
        pass
        '''
        vertices = deepcopy(self.vertices)
        self.vertices = []
        for ax, ay in vertices:
            self.vertices.append(self.rotated_about(ax, ay, IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2,
                                 math.radians(angle)))
        '''


class Circle(Shape):
    def __init__(self):
        super().__init__()

    def generate(self, vertices=True, color=True):
        if color:
            self.color = generate_color(self.mode, self.transparency)
        if vertices:
            """
            self.vertices = []  # To handle mutation.
            curr_point = generate_point(IMAGE_WIDTH, IMAGE_HEIGHT)
            self.vertices.append(curr_point)
            d = randrange(0, POLY_DIAMETER + 1)
            d = [d, d]
            self.vertices.append([sum(x) for x in zip(curr_point, d)])
            """
            self.vertices = self.init_points()
        self.coord = np.mean(np.asarray(self.vertices), axis=0)

    def init_points(self):
        # TODO: penalty for large circles
        # point for circle
        a = randrange(1, int(POLY_DIAMETER / 2))
        b = randrange(1, int(POLY_DIAMETER / 2))
        x = randrange(0, IMAGE_WIDTH)
        y = randrange(0, IMAGE_HEIGHT)
        return [(x - a, y - b), (x + a, y + b)]

    def draw(self, draw):
        draw.ellipse(make_tuple(self.vertices), fill=tuple(self.color), outline=tuple(self.color))


class Polygon(Shape):
    def __init__(self):
        super().__init__()

    def generate(self, vertices=True, color=True):
        if color:
            self.color = generate_color(self.mode, self.transparency)
        if vertices:
            self.vertices = []  # To handle mutation.
            curr_point = generate_point(IMAGE_WIDTH - POLY_DIAMETER, IMAGE_HEIGHT - POLY_DIAMETER)
            for i in range(randrange(MIN_VERTICES, MAX_VERTICES + 1)):
                self.vertices.append([sum(x) for x in zip(curr_point,
                                                          generate_point(POLY_DIAMETER, POLY_DIAMETER))])
        self.coord = np.mean(np.asarray(self.vertices), axis=0)

    def draw(self, draw):
        draw.polygon(make_tuple(self.vertices), fill=tuple(self.color), outline=tuple(self.color))


class Genotype:
    def __init__(self):
        self.shapes = []
        self.fitness = float('nan')
        self.image = None
        self.last_rotate = 0  # or 'R'
        self.angle = 0

    def generate(self):
        for i in range(NUMBER_OF_POLYGONS):
            if random() < POLYGON_RATE:
                new_shape = Polygon()
            else:
                new_shape = Circle()
            new_shape.generate()
            bisect.insort(self.shapes, new_shape)
            #self.shapes.append(new_shape)

    def get_fitness(self):
        if np.isnan(self.fitness):
            self.compute_fitness()
        return self.fitness

    def compute_fitness(self):
        if self.image is None:
            self.generate_image()
        if IMAGE_MATRIX is not None:
            self.fitness = get_image_error(self.image)
        else:
            self.fitness = get_classifier_error(self.image, IMAGE_LABEL)

    def get_image(self):
        if self.image is None:
            self.generate_image()

        return self.image

    def generate_image(self):
        draw_image = Image.new('RGBA', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255, 0))
        draw = ImageDraw.Draw(draw_image)

        for shape in self.shapes:
            shape.draw(draw)
        draw_image = draw_image.rotate(self.angle)
        image = Image.new('RGB', draw_image.size, (255,) * 3)
        image.paste(draw_image, mask=draw_image)
        self.image = image

    def mutate(self):
        # for polygon in self.shapes:
        #    if random() < PROBABILITY_MUTATION:
        #        polygon.mutate()
        idx = randrange(0, NUMBER_OF_POLYGONS)
        shape = self.shapes[idx]
        self.shapes.__delitem__(idx)
        shape.mutate()
        bisect.insort(self.shapes, shape)
        #self.shapes[randrange(0, NUMBER_OF_POLYGONS)].mutate()
        self.fitness = float('nan')  # Resetting fitness since the genotype has been mutated.
        self.image = None

    def rotate(self):

        if self.last_rotate == 0.0:
            self.angle = uniform(5, 30) if random() < 0.5 else uniform(-30, -5)
            self.last_rotate = self.angle
        else:
            self.angle = -self.last_rotate
            self.last_rotate = 0.0

        self.fitness = float('nan')
        self.image = None


class Pixeltype(Genotype):
    def __init__(self):
        super().__init__()

    def generate(self):
        if IMAGE_MODE == 'L':
            shapes = np.random.randint(0, 256, (IMAGE_WIDTH, IMAGE_HEIGHT, 1), dtype='int32')
            self.shapes = np.tile(shapes, (1, 1, 3))
        else:
            self.shapes = np.random.randint(0, 256, (IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype='int32')

    def generate_image(self):
        self.image = Image.fromarray(np.array(self.shapes, dtype='uint8'), 'RGB')

    def mutate(self):
        x_idx, y_idx = randrange(0, IMAGE_WIDTH), randrange(0, IMAGE_HEIGHT)
        self.shapes[x_idx, y_idx] = np.random.randint(0, 256, (3, ), dtype='int32')
        self.fitness = float('nan')  # Resetting fitness since the genotype has been mutated.
        self.image = None


class Population:
    def __init__(self):
        self.genotypes = []

    def generate_initial(self):
        ancestor = Genotype()
        ancestor.generate()
        for i in range(POPULATION_SIZE):
            new_member = deepcopy(ancestor)
            for j in range(ELITISM_NUMBER):
                new_member.mutate()
            self.genotypes.append(new_member)

    def select_parents(self):  # Sampling Parents with high fitness
        fitness_list = np.array(self.get_fitness_list())
        fitness_list[fitness_list <= 0] = 1e-100
        fitness_list[np.isnan(fitness_list)] = 1e-100
        fitness_list = fitness_list / sum(fitness_list)
        genotype = np.array(self.genotypes)
        parents = genotype[np.random.choice(len(self.genotypes),
                                            NUMBER_OF_PARENTS,
                                            replace=False, p=fitness_list)]
        return list(parents)

    def compute_total_fitness(self):
        f = lambda geno: sum([member.get_fitness() for member in geno])
        return f(self.genotypes)

    def reproduce(self):
        parents = self.select_parents()
        idx = np.argsort(np.array(self.get_fitness_list()))
        for i in range(ELITISM_NUMBER):
            child = deepcopy(self.genotypes[idx[i]])
            child.mutate()
            self.genotypes.append(child)
        self.crossover(parents)

    def crossover(self, parents):
        shuffle(parents)
        for i in range(0, NUMBER_OF_PARENTS, 2):
            child_1, child_2 = self.generate_crossover_children(parents[i], parents[i + 1])
            self.genotypes.append(child_1)
            self.genotypes.append(child_2)

    def generate_crossover_children(self, parent_1, parent_2):  # Single Point Crossover
        crossover_point = randrange(1, NUMBER_OF_POLYGONS - 1)
        child_1, child_2 = Genotype(), Genotype()
        f = lambda par, child, i: child.shapes.append(deepcopy(par.shapes[i]))
        for i in range(crossover_point):
            f(parent_1, child_1, i)
            f(parent_2, child_2, i)
        for i in range(crossover_point, NUMBER_OF_POLYGONS):
            f(parent_1, child_2, i)
            f(parent_2, child_1, i)
        child_1.mutate()
        child_2.mutate()

        return child_1, child_2

    def mutate(self):
        for genotype in self.genotypes:
            if random() < PROBABILITY_MUTATION:
                genotype.mutate()

    def elitism(self):
        self.genotypes.sort(key=lambda f: f.get_fitness(), reverse=False)
        self.genotypes = self.genotypes[:ELITISM_NUMBER] + sample(self.genotypes[ELITISM_NUMBER:],
                                                                  POPULATION_SIZE - ELITISM_NUMBER)

    def get_subset_sum(self, end, start=0):
        subset_sum, i = 0.0, start
        while i <= end:
            subset_sum += self.genotypes[i].get_fitness()
            i += 1
        return subset_sum

    def get_best(self):
        return np.argmin([g.get_fitness() for g in self.genotypes])

    def get_best_fitness(self):
        return min([g.get_fitness() for g in self.genotypes])

    def get_fitness_list(self):
        return [g.get_fitness() for g in self.genotypes]

    def perturbation(self):
        # challenge the most fit individual in the population
        if IMAGE_MATRIX is None and self.get_best_fitness() < CHALLENGE_SCORE:
            idx = self.get_best()
            self.genotypes[idx].rotate()


def generate_color(mode, transparency):
    if mode == 'L':
        idx = randrange(0, 256)
        color = [idx for i in range(3)]
        if transparency:
            color = color + [randrange(128, 256)]
        else:
            color = color + [255]
        return color
    else:
        color = [randrange(0, 256) for i in range(3)]
        if transparency:
            color = color + [randrange(128, 256)]
        else:
            color = color + [255]
        return color


def generate_point(x_max, y_max):  # Include offset.
    x, y = randrange(0, x_max + 1), randrange(0, y_max + 1)
    return [x, y]


def make_tuple(vertices):
    return [tuple(vertex) for vertex in vertices]


def get_image_error(image1):
    current_array = np.asarray(image1, dtype='int32')
    error = abs(IMAGE_MATRIX - current_array).sum()

    return error


def get_classifier_error(image1, label):
    # adapt_util.get_error(image1, label)
    return -predict(image1, label=label)


def initialize_global_vars(image):
    global INPUT_IMAGE, IMAGE_MATRIX, \
        IMAGE_WIDTH, IMAGE_HEIGHT, \
        POLY_DIAMETER, IMAGE_MODE
    INPUT_IMAGE = image
    IMAGE_WIDTH, IMAGE_HEIGHT = image.size
    IMAGE_MATRIX = []
    POLY_DIAMETER = IMAGE_WIDTH * POLY_DIAMETER if IMAGE_WIDTH < IMAGE_HEIGHT else IMAGE_HEIGHT * POLY_DIAMETER
    POLY_DIAMETER = int(POLY_DIAMETER)
    IMAGE_MATRIX = np.asarray(INPUT_IMAGE, dtype='int32')
    IMAGE_MODE = INPUT_IMAGE.mode


def initialize_genetic_vars(image_size=(80, 80), img_label=0, img_mode='L'):
    global INPUT_IMAGE, IMAGE_WIDTH, \
        IMAGE_HEIGHT, IMAGE_MATRIX, \
        POLY_DIAMETER, IMAGE_LABEL, IMAGE_MODE

    INPUT_IMAGE = None
    IMAGE_WIDTH, IMAGE_HEIGHT = image_size
    IMAGE_MATRIX = None
    POLY_DIAMETER = IMAGE_WIDTH * POLY_DIAMETER if IMAGE_WIDTH < IMAGE_HEIGHT else IMAGE_HEIGHT * POLY_DIAMETER
    POLY_DIAMETER = int(POLY_DIAMETER)
    IMAGE_LABEL = img_label
    IMAGE_MODE = img_mode

def initialize_classifier():
    global CLASSIFIER
    CLASSIFIER = adapt_util.Classifier()
    CLASSIFIER.pretraining()


def set_variables(**kwargs):
    glob_dict = globals()
    for key in kwargs:
        if glob_dict.__contains__(key):
            glob_dict[key] = kwargs[key]
