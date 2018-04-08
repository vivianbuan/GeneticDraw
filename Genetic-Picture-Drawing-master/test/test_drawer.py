from gendraw import GeneticDrawer
from PIL import Image
import os


__author__ = 'Ilya'


if __name__ == '__main__':
    pic_path = os.path.join(os.path.abspath('../'), 'pic')
    pic_name = 'Katherine.png'
    pic_filename = os.path.join(pic_path, pic_name)
    pic = Image.open(pic_filename)
    pic = pic.convert('RGB')
    # drawer = GeneticDrawer(pic, n_circles=256, n_generations=10e6)
    save_path = os.path.join(os.path.abspath('../'), 'save')
    drawer = GeneticDrawer(pic, n_circles=256, n_generations=1e5, path=save_path)
    drawer.draw()
