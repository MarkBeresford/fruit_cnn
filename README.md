# fruit_cnn

The data used in this model can be found [here](https://www.kaggle.com/moltean/fruits).

I achieved 96% accuracy on the test set.

For easy reuse the downloaded data should be put in the src folder. 

The fruit_cnn.py script can be run from the command line, to train a model using the fruit-360 dataset (placed in the src folder):
```
python fruit_cnn.py train
``` 
for validation (testing against the validation data set): 
```
python fruit_cnn.py train
```
and to use the model to predict which fruit is in an image: 
```
python fruit_cnn.py predict <image_dir> 
```
For predictions the image should be a .jpg file. And the image dir argument should point to the directory the image is in. There can be more then one image in the directory, the model will make a prediction for each .jpg file in the directory.

I also found [this](http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/) resource very useful for understanding how to build a cnn.
