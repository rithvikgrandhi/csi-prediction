# CSI Prediction using iTransformer

This project aims to predict Channel State Information (CSI) using a customized Transformer architecture known as the `iTransformer`. The `iTransformer` model is adapted for time series forecasting, especially for high-dimensional data sets as commonly found in communication systems.

## Overview

Channel State Information (CSI) pertains to the properties of communication channels in wireless communication. Knowing CSI aids in adapting transmissions to current channel conditions, which is crucial for optimizing the performance of wireless networks. The iTransformer model, with its attention mechanism, is particularly well-suited to handle the temporal dependencies and feature correlations inherent in CSI data.

## Requirements

- Python 3.x
- PyTorch
- Numpy
- Matplotlib
- Scikit-learn
- Pandas

## Setup

To set up the project environment, install the required packages using pip:

```bash
pip install torch numpy matplotlib scikit-learn pandas
```

## Data Preprocessing

The data is preprocessed to transform raw CSI data into a suitable format for the model. This involves creating sequences of past channel states to predict future states. The preprocessing function generates input-target pairs from a sliding window over the time series data:

```python
import numpy as np
import torch

def preprocess_CSI_data(data, lookback_len=10):
    num_UEs, num_timesteps, num_features = data.shape
    input_list = []
    target_list = []

    for ue in range(num_UEs):
        ue_data = data[ue]
        for i in range(num_timesteps - lookback_len):
            input_seq = ue_data[i:i + lookback_len]
            target_seq = ue_data[i + lookback_len]
            input_list.append(input_seq)
            target_list.append(target_seq)

    inputs = torch.tensor(input_list, dtype=torch.float32)
    targets = torch.tensor(target_list, dtype=torch.float32)
    return inputs, targets
```

## Model

The iTransformer model is configured with specific parameters tailored for the prediction of CSI:

```python
from iTransformer import iTransformer

model = iTransformer(
    num_variates=256,
    lookback_len=10,
    dim=256,
    depth=6,
    heads=8,
    dim_head=64,
    pred_length=1,
    num_tokens_per_variate=1,
    use_reversible_instance_norm=True
)
```

## Mathematical Equation

The iTransformer model relies on the self-attention mechanism, where the attention score is calculated as follows:

\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]

Where:
- \( Q \), \( K \), and \( V \) are the query, key, and value matrices derived from the input.
- \( d_k \) is the dimension of the keys.

## Evaluation

To evaluate the model, metrics such as MSE, RMSE, MAE, and R-squared are computed:

```python
import torch.nn.functional as F
from sklearn.metrics import r2_score
from math import sqrt

def evaluate_model(model, inputs, targets):
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).squeeze()
        mse = F.mse_loss(predictions, targets).item()
        rmse = sqrt(mse)
        mae = F.l1_loss(predictions, targets).item()
        r2 = r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
    return mse, rmse, mae, r2
```

## Conclusion

The iTransformer model, with its ability to handle high-dimensional time series data, is used to predict future channel states effectively, providing essential insights for optimizing communication channels in wireless networks.
