# HypoGuardian AI: Dataset Documentation & Data Access Guide

## 1. Primary Research Dataset: OhioT1DM

### 1.1 Overview
The **OhioT1DM dataset** is an internationally recognized continuous glucose monitoring benchmark developed by the School of Electrical Engineering and Computer Science at Ohio University (Marling & Bunescu). It contains 8 weeks of continuous data for individuals with Type 1 Diabetes on insulin pump therapy.

### 1.2 Data Attributes Included in OhioT1DM
- **Continuous Glucose Monitoring (CGM)**: Interstitial glucose sampled every 5 minutes via Dexcom or Medtronic sensors.
- **Basal / Bolus Insulin**: Insulin infusion rates and mealtime correction boluses.
- **Self-Reported Meals**: Carbohydrate intake estimates.
- **Biometric Sensor Traces**: Heart rate, galvanic skin response (GSR), skin temperature, and step counts (via Empatica Embrace / Basis Peak).

### 1.3 Authorized Access & Acquisition Instructions
Because the OhioT1DM dataset contains sensitive clinical health records from real patients, it is **not public domain** and cannot be hosted directly in open Git repositories or downloaded without credentialing.

**Official Application Process:**
1. Navigate to the official Ohio University T1D portal: `http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html`
2. Submit a formal academic request specifying:
   - Name of Student Researcher and Faculty Project Advisor
   - Academic Institution / Department
   - Intended Research Purpose (e.g., Undergraduate B.Tech Machine Learning Capstone)
3. Execute the formal **Data Use Agreement (DUA)** adhering to HIPAA compliance.
4. Upon approval, authorized researchers receive credentials to download the XML archives (`559-ws-training.xml`, `563-ws-training.xml`, etc.).
5. Place the downloaded XML files in `HypoGuardian-AI/data/raw/`. The built-in parser in `src/data_loader.py` (`parse_ohiot1dm_xml`) will automatically ingest and normalize the data.

---

## 2. Synthetic Benchmark Cohort (Included in Project)

To enable immediate local testing, automated testing, and model training without violating clinical data privacy laws, HypoGuardian AI includes a high-fidelity synthetic data generator (`src/generate_synthetic_data.py`).

### 2.1 Physiological Simulation Mechanics
The synthetic generator implements an **Ornstein-Uhlenbeck stochastic mean-reversion process** combined with physiological dynamics:
1. **Circadian Glycemic Baseline**: Sinusoidal oscillation reflecting diurnal cortisol/growth-hormone cycles (amplitude $\pm 10\text{ mg/dL}$, nadir at 04:00 AM).
2. **Postprandial Carbohydrate Spikes**:
   - Breakfast (08:00): Peak $+35\text{ mg/dL}$
   - Lunch (13:00): Peak $+40\text{ mg/dL}$
   - Dinner (19:30): Peak $+45\text{ mg/dL}$
3. **Hypoglycemic Excursions**: Periodic nocturnal insulin-glucose mismatches and physical activity dips causing glucose to drop below $70\text{ mg/dL}$ ($50 - 68\text{ mg/dL}$) for 30–60 minutes.
4. **Sensor Interstitial White Noise**: $\mathcal{N}(0, 2.5\text{ mg/dL})$ Gaussian noise modeling Dexcom G6 sensor variance.

### 2.2 Dataset Files Included
- **`data/raw/synthetic_cgm_cohort.csv`**:
  - Multi-patient cohort: 3 virtual patients (`SYNTH_P01`, `SYNTH_P02`, `SYNTH_P03`), 7 days each ($6,048$ total readings).
  - Used for training, validation, and held-out test evaluation.
- **`data/sample/synthetic_cgm_sample.csv`**:
  - Single-patient 24-hour trace ($288$ readings).
  - Used for quick Streamlit drag-and-drop upload demonstration and testing.

---

## 3. Data Schema Specification

Every CSV processed by HypoGuardian AI adheres to the following specification:

| Field Name | Type | Format / Allowed Values | Required? | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | String / Datetime | `YYYY-MM-DD HH:MM:SS` | **Yes** | Date and time of the interstitial glucose reading. |
| `glucose` | Float | $20.0 \le \text{glucose} \le 500.0$ | **Yes** | Blood glucose concentration in **$\text{mg/dL}$**. |
| `patient_id` | String | Alphanumeric (e.g. `P01`) | Optional | Patient identifier (defaults to `PATIENT_01` if omitted). |
| `source` | String | Free text | Optional | Dataset provenance tag. |
