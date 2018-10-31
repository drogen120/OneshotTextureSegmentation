### Oneshot TextureSegmentation
An implementation of Oneshot TextureSegmentation
https://arxiv.org/abs/1807.02654

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

![input image a](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/image_1530.jpg)
![input image a texture](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/texture_1530.jpg)


![predict mask](https://raw.githubusercontent.com/drogen120/OneshotTextureSegmentation/master/results/image_pred_1530.jpg)
