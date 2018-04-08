from PIL import ImageDraw
import random


__author__ = 'Ilya'


class Circle:
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

