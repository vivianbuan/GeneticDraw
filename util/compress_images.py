from skimage.io import imsave, imread
from skimage.transform import resize
from os import listdir
from os.path import isfile, join
import rawpy

mypath = 'D:\\360Sync\OneDrive\Berkeley\MachineLearning\Spring2018\Project\Images'
savepath = 'D:\\360Sync\OneDrive\Berkeley\MachineLearning\Spring2018\Project\Images'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for _, filename in enumerate(onlyfiles):
    pure_name = filename[:-4]
    try:
        img = rawpy.imread(mypath + '\\' + filename).postprocess()
    except:
        img = imread(mypath + '\\' + filename)
    height, width, _ = img.shape
    start = int(width/2-height/2)
    # img = img[:, start:start+height]
    img = resize(img, (100, 100), mode='reflect')
    imsave(savepath + '\\' + pure_name + '_compressed.png', img)
