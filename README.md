### Oneshot TextureSegmentation
An implementation of Oneshot TextureSegmentation
https://arxiv.org/abs/1807.02654

### Environment
Python 3.5

Tensorflow 1.9

Keras 2.2

### How to use

Download dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html#citation

The dataset should follow the following folder structure:

home

|-- Dataset

| |-- dtd

| | |-- images

| | |-- imdb

| | |-- labels

run python train.py to train the model.

### Results

Example 1


![input image a](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/image_1530.jpg)
![input image a texture](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/texture_1530.jpg)
![predict mask](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/image_pred_1530.jpg)


Example 2


![input image a](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/image_1830.jpg)
![input image a texture](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/texture_1830.jpg)
![predict mask](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/image_pred_1830.jpg)


Example 3


![input image a](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/image_1835.jpg)
![input image a texture](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/texture_1835.jpg)
![predict mask](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/image_pred_1835.jpg)

### Model Weights Downloads

https://drive.google.com/file/d/1HOyIaBOpYyZeAFJOLjvcOaddSbCSVMig/view?usp=sharing
