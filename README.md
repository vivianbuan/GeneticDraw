# Evolutionary Painting: Use genetic algorithm and convolutional neural network to create artworks。

The GeneticDraw is a model designed for automatically drawing images given a label using genetic algorithm and CNN classifier. This is the result of gruaduation student final project of [Berkeley EECS 189/289A course](http://www.eecs189.org/).

If you're interested, checkout our [final report](docs/289_Final_Report.pdf) for details of experiment results and discussion.

## Instroduction
In this project we investigate the ability of genetic algorithm to draw original painting. We design a selection, crossover, and mutation model using 50 shapes of ellipses and polygons to draw MNIST-like digits given a specific target label. A MNIST classifier with three convolutional layers and one fully connected layer was used to evaluate the generated images by computing the Kullback–Leibler divergence from the predicted probabilities to the one hot vector of the desired label. The final results we generate are human decipherable, as well as incorporating nice art effects.

The genetic algorithm part of our model is modified from [vishaldotkhanna/polygonGA](https://github.com/vishaldotkhanna/polygonGA). <br/>
The MNIST classifier is modified from [golbin/TensorFlow-MNIST](https://github.com/golbin/TensorFlow-MNIST)

## Model
<p align="center">
<img title="GeneticDraw Model" src="/docs/img/model.jpg">
</p>

## Result
<img align="left" width="500" title="GeneticDraw Model" src="/docs/img/Final.png">

The results of using our model to draw MNIST-like digits given target label 0-9. The results are not perfect but are human decipherable. <br/>

Notice that we are not trying to reconstruct perfect imageslike those are done with generative adversarial network (GAN)  whose  idea  is  also  derived  from  biology,  but  toexpand  the  idea  of  evolutionary  painting  associated  with  theknowledge in machine learning.

<br/><br/>

## Usage
Run ```main.py``` in ```polygonGA-modified/``` to start drawing. To specify the target digit to draw, change ```img_label``` on line ```165```. <br/> 

[TensorFlow-MNIST-modified/models/has-unknown/drop-out/](TensorFlow-MNIST-modified/models/has-unknown/drop-out/) folder contains pretrained MNIST classifier with images of one label been dropt out and an unknown label related to randomly generated images introduced. If you're Pycharm user, remember to add the model to library.<br/>

To train your own model, run ```train.py``` in ```TensorFlow-MNIST-modified/```. You can choose whether or not to include drop out and unknown by setting the flags accordingly.

Our model can be extended to different image classifier. We experimented with VGG16 but didn't result in readable images. For more details of our experiments and decision choice, refer to our [final report](docs/289_Final_Report.pdf).

## Authors
* [Weiran(Vivian) Liu](https://github.com/vivianbuan)
* [Huanjie Sheng](https://github.com/david190810)
