# Soft Biometry Using Convolutional Networks
This is the repository for our paper titled "Soft Biometry Using Convolutional Networks: Human Attributes Classification and Object Carrying Detection".

![architecture cnn](https://github.com/CaioSilva1/MArc/blob/master/architecture/architecture.jpg)

## Data 
The data used in this project comes from the source: 
* The [PETA dataset](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html), consists of 19000 images.

## Code 
The code is divided as follows: 
* The [MArc_TensorFlow.py](https://github.com/CaioSilva1/MArc/blob/master/MArc_TensorFlow.py) python file contains the code to run experiment.
* The [MArc_TensorFlow_version_test.py](https://github.com/CaioSilva1/MArc/blob/master/MArc_TensorFlow_version_test.py) python file contains the code to run experiment of test.

## Pre-requisites
The code uses Tensorflow 2.0.

* [numpy](http://www.numpy.org/)  
* [scipy](https://www.scipy.org/)  
* [tensorflow-gpu](https://www.tensorflow.org/)  

## Results
The graph shows that the MArc model learns faster, that is, with 300 training epochs, our model has already learned and saturated inaccuracy, while the CNN-OM model, even with 1000 training epochs, it is still learning.
![accuracy curve MArc model](https://github.com/CaioSilva1/MArc/blob/master/performance/performance.png)


Our results in the paper showed that a proposed CNN approach (MArc) has provided promising results and performs best for the soft biometry classification.

The following table contains the averaged accuracy over 30 runs of the MArc model on the PETA dataset, with the standard deviation between parentheses. 

| Gender                         | 0.003           | 0.01            |
|--------------------------------|-----------------|-----------------|
| Male                           | 77.91(2.75)     | **82.45**(3.14) |
| Female                         | **82.85**(2.77) | 78.53(3.76)     |

| Upper Clothes                  | 0.003           | 0.01            |
|--------------------------------|-----------------|-----------------|
| Short                          | **74.16**(2.19) | 74.11(2.63)     |
| Long                           | 71.46(2.82)     | **73.18**(2.16) |

| Lower Clothes                  | 0.003           | 0.01            |
|--------------------------------|-----------------|-----------------|
| Shorts                         | **85.91**(2.68) | 83.26(12.33)    |
| Pants                          | 75.45(3.06)     | **80.08**(2.74) |

| Carrying object                | CNN-OM-BN       | MArc            |
|--------------------------------|-----------------|-----------------|
| Nothing                        | 50.30(1.82)     | **55.76**(1.23) |
| Something                      | **65.53**(1.68) | 58.83(1.37)     |

These results should give an insight of deep learning for soft biometrics classification in surveillance systems applications.