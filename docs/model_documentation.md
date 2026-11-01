# HypoGuardian AI: Deep Learning Model Documentation

## 1. Formulation of the Prediction Task

The core objective of HypoGuardian AI is to provide a clinically meaningful **30-minute early warning** of impending hypoglycemia.

### 1.1 Mathematical Definition
Let $G_t$ denote the interstitial blood glucose observation at discrete timestamp $t$, sampled at approximately 5-minute intervals ($\Delta t \approx 5\text{ min}$).

- **Historical Input Window ($X_t$)**:
  $$X_t = [G_{t-11}, G_{t-10}, \dots, G_t] \in \mathbb{R}^{12 \times 1}$$
  This captures 12 sequential readings across 11 intervals ($\approx 55\text{ minutes}$ of physiological history).

- **Forecast Horizon ($H$)**:
  $$H_t = [G_{t+1}, G_{t+2}, \dots, G_{t+6}]$$
  This spans the subsequent 30 minutes.

- **Binary Target Label ($y_t$)**:
  $$y_t = \begin{cases} 
  1, & \text{if } \min(G_{t+1}, \dots, G_{t+6}) < 70.0\text{ mg/dL} \\ 
  0, & \text{otherwise} 
  \end{cases}$$

---

## 2. Model Architecture

HypoGuardian AI employs a Long Short-Term Memory (LSTM) Recurrent Neural Network. LSTMs are uniquely capable of overcoming the vanishing gradient problem inherent in vanilla RNNs, allowing the network to retain both long-term baseline trends and acute short-term drop rates (glucose velocity $\frac{dG}{dt}$ and acceleration $\frac{d^2G}{dt^2}$).

```
Input Layer:           (Batch Size, 12, 1)
   │
LSTM Layer:            64 Units, Tanh Activation, Return Sequences = False
   │
Dropout Layer:         Rate = 0.30
   │
Dense Layer:           32 Units, ReLU Activation
   │
Dropout Layer:         Rate = 0.20
   │
Output Dense Layer:    1 Unit, Sigmoid Activation -> Probability p in [0.0, 1.0]
```

### 2.1 The LSTM Memory Cell Mechanics
At each step $\tau \in \{1, \dots, 12\}$, the LSTM cell updates its candidate hidden state:
- **Forget Gate ($f_\tau$)**: Decides what percentage of the previous cell state $C_{\tau-1}$ to discard:
  $$f_\tau = \sigma(W_f \cdot [h_{\tau-1}, x_\tau] + b_f)$$
- **Input Gate ($i_\tau$)**: Decides which new glucose information to store into cell state:
  $$i_\tau = \sigma(W_i \cdot [h_{\tau-1}, x_\tau] + b_i)$$
  $$\tilde{C}_\tau = \tanh(W_c \cdot [h_{\tau-1}, x_\tau] + b_c)$$
- **Cell State Update ($C_\tau$)**:
  $$C_\tau = f_\tau * C_{\tau-1} + i_\tau * \tilde{C}_\tau$$
- **Output Gate ($o_\tau$) & Hidden State ($h_\tau$)**:
  $$o_\tau = \sigma(W_o \cdot [h_{\tau-1}, x_\tau] + b_o)$$
  $$h_\tau = o_\tau * \tanh(C_\tau)$$

---

## 3. Training & Optimization Strategy

### 3.1 Loss Function
The model optimizes Binary Cross-Entropy with balanced class weighting:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ w_1 \cdot y_i \log(\hat{y}_i) + w_0 \cdot (1 - y_i) \log(1 - \hat{y}_i) \right]$$
Where $w_1$ and $w_0$ are computed from the training split via:
$$w_c = \frac{N_{\text{samples}}}{2 \cdot N_c}$$

### 3.2 Hyperparameters & Callbacks
- **Optimizer**: Adam ($\text{learning rate} = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$)
- **Batch Size**: 32
- **Maximum Epochs**: 35
- **Early Stopping**: Restores best model weights if `val_loss` does not improve for 7 consecutive epochs.
- **Learning Rate Reduction**: Multiplies learning rate by $0.5$ if `val_loss` plateaus for 3 epochs.

---

## 4. Evaluation Strategy & Trade-Offs

### 4.1 Chronological Split Strategy
Random shuffling of time-series samples leaks adjacent rolling windows between train and test partitions. To guarantee zero data leakage:
- **Training Set ($70\%$)**: Earliest chronological partition.
- **Validation Set ($15\%$)**: Intermediate period used for early stopping and threshold selection.
- **Held-Out Test Set ($15\%$)**: Latest chronological period, kept strictly untouched until final evaluation.

### 4.2 The Clinical Sensitivity vs. Specificity Trade-Off
In hypoglycemia monitoring:
- **High Sensitivity (Recall)** is critical because missed hypoglycemic episodes can lead to loss of consciousness or diabetic coma.
- **High Specificity** is critical to avoid "alert fatigue" — where users disable warning systems because of too many false alarms.
- The default decision threshold ($\theta = 0.50$) can be tuned interactively in the Streamlit application to accommodate patient-specific risk profiles.
