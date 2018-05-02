# draw random houndstooth picasso art (4)
from random import randrange
import PIL
import random
from copy import deepcopy
import numpy
import os
from PIL import Image, ImageDraw
import numpy as np

class Shape:
    def __init__(self, borders=None, mode='L'):
        pass

class Polygon(Shape):
    """ Basic element for picture drawing """

    def __init__(self, borders=None, mode='L'):
        self.borders = borders
        self.mode = mode

        # color of circle
        if self.mode == 'L':
            self.color = random.randint(0, 255)
        elif self.mode == 'RGB':
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.color = (r, g, b)
        else:
            raise ValueError

        self.generate()

    def generate_point(self, x_max, y_max):  # Include offset.
        x, y = randrange(0, int(x_max) + 1), randrange(0, int(y_max) + 1)
        return [x, y]

    def generate(self):
        self.points = []  # To handle mutation.
        curr_point = self.generate_point(self.borders[0] * 0.8, self.borders[1] * 0.8)
        for i in range(randrange(3, 5 + 1)):
            self.points.append([sum(x) for x in zip(curr_point,
                                                      self.generate_point(self.borders[0] * 0.2, self.borders[1] * 0.2))])

    def make_tuple(self, vertices):
        return [tuple(vertex) for vertex in vertices]

    def draw(self, image):
        drawer = ImageDraw.ImageDraw(image)
        drawer.polygon(self.make_tuple(self.points), fill=self.color)
        return image


class Circle(Shape):
    """ Basic element for picture drawing """
    def __init__(self, borders=None, mode='L'):
        self.borders = borders
        self.mode = mode

        # color of circle
        if self.mode == 'L':
            self.color = random.randint(0, 255)
        elif self.mode == 'RGB':
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.color = (r, g, b)
        else:
            raise ValueError

        self.points = self._init_points()

    def _init_points(self):
        # TODO: penalty for large circles
        # point for circle
        r = random.randint(1, min(self.borders))/10
        x = random.randint(0, self.borders[0])
        y = random.randint(0, self.borders[1])
        return [x-r, y-r, x+r, y+r]

    def draw(self, image):
        drawer = ImageDraw.ImageDraw(image)
        drawer.ellipse(self.points, fill=self.color)
        return image


class GeneticDrawer:
    """ Class that draws image based on circles """
    def __init__(self, pattern_picture, n_circles=100, n_generations=100, path=None, savefreq=1000):
        self.pattern_picture = pattern_picture
        self.n_generations = n_generations
        self.pattern_array = numpy.asarray(self.pattern_picture, dtype='int32')
        self.n_circles = n_circles
        self.mode = self.pattern_picture.mode
        self.size = self.pattern_picture.size
        self.circles = self.get_shapes()
        self.image = None
        self.path = path
        self.savefreq = savefreq

    def get_shapes(self):
        shapes = []
        for _ in range(self.n_circles):
            if np.random.rand() < 0.5:
                shapes.append(Polygon(self.size, self.mode))
            else:
                shapes.append(Circle(self.size, self.mode))
        return shapes


    def mutate_circles(self):
        # choose circle that would be mutated
        new_circles = deepcopy(self.circles)
        rand_index = random.randint(0, self.n_circles - 1)
        new_circles[rand_index] = Circle(self.size, self.mode)
        return new_circles

    def show(self):
        if not self.image:
            raise NotImplementedError
        self.image.show()

    def draw_image(self, circles):
        # draw image
        image = Image.new(self.mode, self.size, color='white')
        for circle in circles:
            circle.draw(image)
        return image

    def draw(self):
        self.image = self.draw_image(self.circles)
        return self.image

    def reset(self, n_circles=100, n_generations=100, path=None, savefreq=1000):
        self.n_generations = n_generations
        self.pattern_array = numpy.asarray(self.pattern_picture, dtype='int32')
        self.n_circles = n_circles
        self.mode = self.pattern_picture.mode
        self.size = self.pattern_picture.size
        self.circles = self.get_shapes()
        self.image = None
        self.path = path
        self.savefreq = savefreq

    def random_circles(self, n_img=10, save_img=False, save_data='train'):

        all_imgs = []
        data = []
        savepath = os.path.join(os.path.abspath('./'), 'save')
        for i in range(n_img):
            img = self.draw()
            all_imgs.append(img)
            if save_img:
                img.save(os.path.join(savepath, 'test' + str(i) + '.png'))
            img_data = np.array(img).flatten() / 255
            print('max is ', np.max(img_data))
            data.append(img_data)
            self.reset()
        data_shape = (-1, ) + self.size + (1, )
        data = np.array(np.vstack(data), dtype=float).reshape(data_shape)
        if save_data == 'train' or save_data == 'test':
            np.save(os.path.join(savepath, 'data_' + str(save_data) + '.npy'), data)
        return all_imgs, data


if __name__ == '__main__':
    # set an image template
    # use 'L' for black and white image
    # use 'RGB' for color image
    test_img = Image.new('L', (28, 28), color='white')
    # initialize GeneticDrawer
    gd = GeneticDrawer(test_img)
    all_imgs, data = gd.random_circles(n_img=5500, save_data='train')

    gd = GeneticDrawer(test_img)
    all_imgs, data = gd.random_circles(n_img=1000, save_data='test')

    # np.save(, data)
    # set save=False so images are not saved
    # all_imgs is a list of random PIL images generated
    # data is a numpy float array with dimension n*d
    # n is the number of image and d is the size of the image




