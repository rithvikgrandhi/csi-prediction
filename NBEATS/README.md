# CSI Prediction with N-BEATS

This repository contains the implementation of the N-BEATS model for predicting Channel State Information (CSI) in communication systems. N-BEATS is a deep learning model designed for time series forecasting that is purely based on forward and backward residual links and a very deep stack of fully connected layers.

## Introduction

Channel State Information (CSI) characterizes how a signal propagates from the transmitter to the receiver and represents the combined effect of, for example, scattering, fading, and power decay with distance. The N-BEATS model, known for its robustness in various forecasting tasks, provides an innovative approach to predict CSI by decomposing the future into a series of basis functions that are learned from the data.

## N-BEATS Overview

N-BEATS is a forecast-based model that eschews traditional recurrent layers and instead utilizes a series of fully connected networks and backward and forward residual links. It operates in two modes: interpretable and generic. In the interpretable mode, each block of the network focuses on a specific aspect of the series (trend, seasonality), while the generic mode does not enforce any predefined interpretability constraints.

### Mathematical Representation

**Forecasting Equation:**
The model forecasts future values by iteratively predicting residuals and adding back to the input:

`y_hat(t+1) = g_theta(x_t) + x_t`

where `x_t` is the input at time `t`, `y_hat(t+1)` is the forecasted output, and `g_theta` is a function representing the N-BEATS model with parameters `theta`.

**Backcasting and Forecasting:**
Each block in N-BEATS produces both a forecast and a backcast, helping the model learn by reconstructing the past data:

- Backcast: `b_t = f_theta(x_t)`
- Forecast: `y_hat(t+1) = g_theta(x_t)`

where `f_theta` is the backcasting function.

## Contributors

- Rohit Viswam
- Rithvik Grandhi
