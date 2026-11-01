# HypoGuardian AI: Comprehensive B.Tech Viva Questions & Answers

This guide is designed for your B.Tech Computer Science project examination, seminar presentation, and viva voce. It covers deep learning theory, time-series sequence modeling, healthcare metrics, clinical continuous glucose monitoring (CGM), and software engineering principles in simple, intuitive language.

---

## Section 1: Clinical Problem & Project Motivation

### Q1: What is hypoglycemia, and why is it dangerous?
**Answer:** Hypoglycemia is a medical condition where blood glucose drops below the safe physiological threshold of **$70\text{ mg/dL}$** ($3.9\text{ mmol/L}$). It is an acute emergency in diabetes management. Because the brain relies almost exclusively on glucose for energy, acute hypoglycemia causes shakiness, dizziness, confusion, seizures, loss of consciousness, diabetic coma, and in severe cases, brain damage or death.

### Q2: What is Continuous Glucose Monitoring (CGM)?
**Answer:** A CGM is a wearable medical sensor inserted under the skin (subcutaneous interstitial fluid) that measures glucose levels automatically every 5 minutes (288 times a day). Popular commercial examples include the Dexcom G6/G7 and Abbott Freestyle Libre.

### Q3: Why is a 30-minute early warning needed instead of a standard alarm?
**Answer:** Contemporary CGM devices trigger threshold alarms only *after* the patient's glucose has already fallen below $70\text{ mg/dL}$. At that point, the patient may already be impaired or symptomatic. A 30-minute early warning gives the patient enough time to consume 15 grams of fast-acting carbohydrates (the clinical "Rule of 15"), allowing glucose to stabilize *before* hypoglycemia occurs.

### Q4: Why is interstitial glucose different from blood glucose?
**Answer:** CGMs measure interstitial glucose (the fluid surrounding tissue cells), not capillary blood glucose directly. There is a natural physiological lag of approximately 5 to 15 minutes between vascular blood glucose and interstitial fluid, which further justifies predictive modeling over reactive alarms.

---

## Section 2: Time-Series Formulation & Preprocessing

### Q5: How is the input sequence structured for the model?
**Answer:** The model accepts a sequence of **12 consecutive readings** sampled at 5-minute intervals. 12 observations across 11 intervals cover approximately **55 minutes** of history. The input shape is `(Batch Size, 12, 1)`.

### Q6: How is the prediction target defined?
**Answer:** It is a binary classification target ($y \in \{0, 1\}$). If **any** glucose reading within the next 30 minutes ($t_{+1}$ to $t_{+6}$) falls below $70\text{ mg/dL}$, $y = 1$ (Hypoglycemia Event). Otherwise, $y = 0$ (Normal/Safe).

### Q7: What is data leakage, and how did you prevent it?
**Answer:** Data leakage occurs when information from the future or test set contaminates the training process. We prevented it through three strict mechanisms:
1. **Chronological Splitting**: We split our dataset temporally ($70\%$ train, $15\%$ val, $15\%$ test) rather than using random shuffling.
2. **Train-Only Scaling**: The `StandardScaler` was fitted strictly on the training partition (`X_train`) and only *applied* (`transform`) to validation and test data.
3. **Patient Boundary Isolation**: Sliding windows never cross from one patient to another.

### Q8: What happens if a patient takes off their CGM sensor for an hour?
**Answer:** This creates a sensor gap ($> 15\text{ minutes}$). If we simply concatenated readings across this gap, the model would falsely treat readings separated by hours as if they were 5 minutes apart. Our sequence builder detects gaps exceeding 15 minutes and discards windows spanning across the gap.

### Q9: Why did you not interpolate missing glucose values over long gaps?
**Answer:** Blood glucose dynamics fluctuate rapidly due to meals and insulin. Interpolating (e.g., linear or spline) over gaps longer than 15-20 minutes fabricates artificial physiological measurements that do not reflect reality, creating dangerous false training labels.

---

## Section 3: Deep Learning & LSTM Architecture

### Q10: Why did you choose an LSTM instead of a standard Artificial Neural Network (ANN) or Random Forest?
**Answer:** Glucose values have temporal dependency — a reading of $85\text{ mg/dL}$ that is dropping at $3\text{ mg/dL/min}$ is dangerous, whereas $85\text{ mg/dL}$ rising at $3\text{ mg/dL/min}$ is safe. Traditional ANNs or tabular models treat features as independent, losing the sequential trend. LSTMs maintain an internal memory state across time steps, naturally capturing velocity and acceleration.

### Q11: How does an LSTM differ from a vanilla Recurrent Neural Network (RNN)?
**Answer:** Vanilla RNNs suffer from the **vanishing gradient problem**: as gradients are backpropagated through time, they shrink exponentially, making it impossible for the RNN to learn long-term dependencies. LSTMs solve this by introducing an explicit **cell state** regulated by three gates:
- **Forget Gate**: Discards irrelevant historical information.
- **Input Gate**: Stores relevant new observations.
- **Output Gate**: Determines which portion of the cell state forms the hidden state.

### Q12: Walk through your model architecture layer by layer.
**Answer:**
1. **Input Layer**: Shape `(12, 1)` (12 time steps, 1 univariate glucose feature).
2. **LSTM Layer**: 64 units extracting temporal trend representations.
3. **Dropout Layer (0.30)**: Randomly zeroes out 30% of neurons during training to prevent overfitting.
4. **Dense Layer (32 units, ReLU)**: Learns non-linear combinations of the extracted temporal features.
5. **Dropout Layer (0.20)**: Additional regularization.
6. **Output Dense Layer (1 unit, Sigmoid)**: Produces an uncalibrated probability $\hat{y} \in [0.0, 1.0]$.

### Q13: What does the Sigmoid activation function do?
**Answer:** The Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ squashes any real-valued number into the range $[0, 1]$, making it the standard activation function for binary classification.

### Q14: What is Dropout and how does it prevent overfitting?
**Answer:** Dropout is a regularization technique where a specified percentage of neuron activations are randomly set to zero during each forward training pass. This prevents neurons from co-adapting (relying too heavily on specific features) and forces the network to learn robust, generalized representations.

---

## Section 4: Machine Learning Metrics & Class Imbalance

### Q15: Why is accuracy a misleading metric in hypoglycemia prediction?
**Answer:** CGM data has **extreme class imbalance**: healthy or stable glucose occupies 98% to 99% of time, while hypoglycemia occupies only 1% to 2%. 
A naive model that predicts "No Hypoglycemia" 100% of the time achieves **99% accuracy**, yet has **0% sensitivity** — it completely fails to save a single patient. Therefore, metrics like Sensitivity (Recall), Specificity, and ROC-AUC are far more informative.

### Q16: Define Sensitivity and Specificity in this context.
**Answer:**
- **Sensitivity (Recall)**: $\frac{TP}{TP + FN}$ — Out of all real hypoglycemia events that occurred, what percentage did our model successfully anticipate?
- **Specificity**: $\frac{TN}{TN + FP}$ — Out of all normal, safe time periods, what percentage did the model correctly identify without sounding a false alarm?

### Q17: What is Precision and why is there a trade-off with Sensitivity?
**Answer:** Precision ($\frac{TP}{TP + FP}$) measures how often an alarm is true when it goes off. If we lower our decision threshold from $0.50$ to $0.20$, we catch more true hypos (Sensitivity increases), but we also trigger more false alarms (Precision decreases).

### Q18: How did you handle class imbalance during model training?
**Answer:** We used **cost-sensitive learning** via class weighting (`compute_class_weight('balanced')`). The loss function penalizes mistakes on rare hypoglycemia events ($y=1$) much more heavily than mistakes on common normal readings ($y=0$), preventing the gradient from being dominated by the majority class.

### Q19: What is ROC-AUC and PR-AUC?
**Answer:**
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Plots True Positive Rate vs. False Positive Rate across all classification thresholds. An AUC of $1.0$ is perfect; $0.5$ is random guessing. Our model achieves **$\approx 0.918$**, indicating strong discriminative ability.
- **PR-AUC (Precision-Recall AUC)**: Focuses exclusively on the minority positive class, providing a stringent assessment for highly imbalanced datasets.

---

## Section 5: Software Engineering, Testing & Deployment

### Q20: Why is the scaler saved using Joblib instead of refitting on new user data?
**Answer:** Refitting a scaler on a new user's 12 uploaded readings is a major data leakage and consistency bug — 12 readings might have a mean of $60$ or $250$, distorting the normalized inputs. The inference engine must use the *exact mean and variance* calculated on the training population.

### Q21: What is the format of the saved model?
**Answer:** We saved the model in the native Keras format: `models/hypo_lstm_model.keras`.

### Q22: What tests did you write, and what do they verify?
**Answer:** We wrote 16 unit tests using `pytest` covering:
1. Data loading and missing column detection.
2. Removal of corrupt and physiologically impossible readings ($<20$ or $>500\text{ mg/dL}$).
3. Deduplication and chronological sorting.
4. Correct $(N, 12, 1)$ sequence shape and target alignment.
5. Discarding sequences with sensor disconnect gaps.
6. Patient boundary separation.
7. Prediction engine safety when model artifacts are missing.
8. Artifact schema validation.

### Q23: Why did you choose Streamlit for the web application?
**Answer:** Streamlit allows building clean, reactive, Python-native medical dashboards that integrate directly with NumPy, Pandas, TensorFlow, and Plotly without the overhead of maintaining separate frontend JavaScript frameworks.

### Q24: What is the purpose of the medical disclaimer in your application?
**Answer:** Machine learning models for health prediction are academic research prototypes. They are not certified by regulatory bodies (such as the FDA or CE) and must never be used for insulin dosing or medical diagnosis. Demonstrating ethical awareness and regulatory limitations is a key engineering requirement in clinical AI.
