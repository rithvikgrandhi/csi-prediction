# CSI Prediction Using XGBoost

This project focuses on predicting Channel State Information (CSI) using the XGBoost algorithm. CSI is crucial in wireless communication as it characterizes the condition of the communication channel which affects the transmission quality and reliability.

## Project Overview

Channel State Information (CSI) represents how a signal propagates from the transmitter to the receiver and describes the combined effect of scattering, fading, and power decay with distance. The knowledge of CSI is essential for optimizing wireless communication system performance, particularly in adapting the transmission rates and selecting beamforming vectors.

We use XGBoost, a powerful machine learning algorithm for regression tasks, to predict CSI values based on historical data. The model is trained using features derived from previous CSI measurements to predict future values, thereby aiding in real-time decision-making regarding signal transmission strategies.

## Methodology

### Data Representation

The CSI data is structured into matrices where each row represents a user equipment (UE) instance with multiple time steps and features:

- **Number of UEs (samples)**: 2100
- **Time steps (features)**: \(398 \times 256\)

### XGBoost Model

XGBoost stands for eXtreme Gradient Boosting, which is an implementation of gradient boosted decision trees designed for speed and performance. It is a powerful approach for building supervised regression models.


#### Model Equation

The model uses the gradient boosting framework where `F(x)` represents the prediction model built by ensemble learning:

```
F(x) = sum of f_k(x) from k=1 to K
```

where:
- `K` is the number of boosting rounds.
- `f_k(x)` represents an individual decision tree.


### Prediction Strategy

1. **Window-based Training**: We employ a sliding window mechanism where:
   - **Input Window Size**: 10 values.
   - **Output Prediction**: Next 4 values.

   The model trains on sequential blocks of 10 time steps to predict the subsequent 4 steps, enhancing the ability to capture temporal dependencies in the data.

2. **Loss Function**: The loss function used in XGBoost for regression tasks is the squared error, calculated as:

```
L(y, y_hat) = (y - y_hat)^2
```

where:
- `y` is the actual value.
- `y_hat` is the predicted value.


## Usage

### Dependencies
- Python 3.x
- XGBoost
- NumPy
- SciKit-Learn


## Evaluation

The model's performance is evaluated using the root mean squared error (RMSE), providing insights into the average magnitude of the prediction errors.

## Conclusion

This project demonstrates the application of XGBoost in predicting CSI, which is vital for the optimization of wireless communication systems. The model aids in understanding complex channel conditions, thus facilitating the dynamic adaptation of system parameters for improved communication quality.
