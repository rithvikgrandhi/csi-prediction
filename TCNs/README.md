# CSI Prediction with Temporal Convolutional Networks (TCNs)

This repository hosts the implementation of Temporal Convolutional Networks (TCNs) for predicting Channel State Information (CSI) in wireless communication systems. TCNs offer an effective alternative to RNNs for sequence modeling tasks, with advantages such as parallelizability, flexible receptive fields, and stable gradients.

## Introduction

Channel State Information (CSI) is crucial for optimizing the performance and reliability of wireless networks. TCNs, characterized by their convolutional architecture, are particularly suited for handling sequence prediction tasks like CSI prediction due to their ability to capture long-term dependencies.

## TCN Model Overview

TCNs use a series of causal convolutions to ensure that the prediction for the current moment is only dependent on past data. They also employ dilated convolutions to increase the receptive field, allowing the network to learn dependencies from data points far back in the sequence.

### Mathematical Representation

**Causal Convolutions:**
Causal convolutions ensure that the model does not violate the temporal order of data. This means that the output at time `t` is calculated only from elements from time `t` or earlier.

**Dilated Convolutions:**
Dilated convolutions introduce gaps in the convolution kernel to cover a larger receptive field without increasing the number of parameters. The dilation factor increases exponentially with the depth of the network. The equation for dilated convolutions is:

`y(t) = (x * f_d)(t) = sum from s=0 to k-1 of f(s) * x(t - ds)`

where `f_d` is a dilated kernel, `k` is the kernel size, `d` is the dilation factor, `x` is the input, and `y` is the output.


## Contributors

- Rohit Viswam
- Rithvik Grandhi
