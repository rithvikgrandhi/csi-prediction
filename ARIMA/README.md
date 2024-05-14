# CSI Prediction Using ARIMA

This project is dedicated to predicting Channel State Information (CSI) using the ARIMA (AutoRegressive Integrated Moving Average) model. CSI is a key metric in wireless communication that characterizes the condition of the communication channel which impacts the quality and reliability of the transmission.

## Project Overview

CSI encapsulates how a signal propagates from the transmitter to the receiver, incorporating effects like scattering, fading, and power decay. Understanding CSI helps optimize the performance of wireless communication systems by adapting transmission rates and selecting optimal beamforming vectors.

We utilize ARIMA, a popular statistical model for time series forecasting, to predict future values of CSI based on historical data. This approach is well-suited for time series data that shows trends, seasonality, or autocorrelation.

## Methodology

### Data Representation

The CSI data is structured into a series where each point represents the CSI at a given time:
- **Number of UEs (samples)**: 2100
- **Time steps per UE**: 398 * 256

### ARIMA Model

ARIMA models are generally denoted as ARIMA(p, d, q), where:
- `p` is the number of lag observations included in the model (lag order).
- `d` is the number of times that the raw observations are differenced (degree of differencing).
- `q` is the size of the moving average window (order of moving average).

#### Model Equation

The ARIMA model is described by the equation:
```
ARIMA Model: X_t = α + β_1 * X_(t-1) + ... + β_p * X_(t-p) + ε_t + θ_1 * ε_(t-1) + ... + θ_q * ε_(t-q)
```
Where:
- `X_t` is the time series at time t.
- `α` is the intercept.
- `β_i` are the parameters of the lagged observations.
- `ε_t` is white noise at time t.
- `θ_i` are the parameters of the moving average.

### Prediction Strategy

1. **Sliding Window Approach**:
   - We use a fixed window of previous time steps to predict the next CSI value, moving the window forward for each new prediction.

2. **Loss Function**:
   - The commonly used loss function in time series forecasting with ARIMA is the Mean Squared Error (MSE), defined as:
   ```
   MSE = (1/n) * sum((y - y_hat)^2)
   ```
   where `y` is the actual value and `y_hat` is the predicted value.

## Usage

### Dependencies
- Python 3.x
- pandas
- statsmodels


## Evaluation

Performance is assessed using the Root Mean Squared Error (RMSE), offering insights into the average magnitude of prediction errors:

```
RMSE = sqrt(MSE)
```


## Contributors

- Rohit Viswam
- Rithvik Grandhi

## Conclusion

This project demonstrates the application of ARIMA for forecasting CSI in wireless communication. This predictive capability aids in proactive system adjustments for enhanced communication quality.
