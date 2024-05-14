
# CSI Prediction with Capsule Networks (CapsNet)

This repository contains the implementation of Capsule Networks (CapsNet) for the prediction of Channel State Information (CSI) in communication systems. CapsNet has shown promising results in tasks that require the capture of hierarchical spatial relationships, making it suitable for CSI prediction where the spatial and temporal features of the channel are critical.

## Introduction

Channel State Information (CSI) represents the properties of a communication channel in a wireless communication environment. Accurate prediction of CSI is crucial for optimizing the performance of wireless communication systems.

Capsule Networks, introduced by Sabour et al., offer a robust architecture for learning spatial hierarchies in data, which is beneficial for modeling the complex dependencies in CSI data.

## CapsNet Overview

Capsule Networks differ from traditional convolutional networks by using groups of neurons (capsules) that encode model parameters in a way that preserves more information about the input data's state. The dynamic routing mechanism between capsules allows the network to learn spatial hierarchies.

### Mathematical Representation

The squashing function used in CapsNet is given by:

\[ v_j = \frac{{\|s_j\|^2}}{{1 + \|s_j\|^2}} \frac{{s_j}}{{\|s_j\|}} \]

where \( v_j \) is the vector output of capsule \( j \) and \( s_j \) is its total input.

The routing algorithm updates the coupling coefficients \( c_{ij} \) based on the agreement between the current output \( v_j \) and the prediction \( \hat{u}_{j|i} \) made by capsule \( i \):

\[ c_{ij} = \frac{{\exp(b_{ij})}}{{\sum_k \exp(b_{ik})}} \]

where \( b_{ij} \) are the log prior probabilities that capsule \( i \) should be coupled to capsule \( j \).


## Contributors

- Rohit Viswam
- Rithvik Grandhi


## Citation

If you find this work useful, please cite it as:

```
@misc{your_csi_capsnet_2024,
  title={CSI Prediction with Capsule Networks},
  author={rohitviswam,rithvikgrandhi},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/rithvikgrandhi/csi-prediction/tree/main/CapsNet}}
}
```
