# Hybrid Vision Analysis for Solar Panel Defects

Welcome to the **Hybrid Vision + Time-Series Analysis** project for Solar Panel Fault Detection. This repository implements an advanced AI system that combines visual inspection (Convolutional Neural Networks) with electrical data analysis (LSTM) to provide a holistic health assessment of solar panels.

---

## 📖 Introduction & Approach

### The Problem
Traditional solar panel inspection relies solely on visual checks (drones/cameras) or purely on electrical monitoring (inverters). Both have blind spots:
*   **Visual-only** misses invisible internal electrical failures (like hotspots behind clean glass).
*   **Electrical-only** detects power loss but cannot pinpoint the physical cause (is it dust? crack? shading?).

### Our Hybrid Approach
We propose a **Multi-Modal Fusion Model** that processes two distinct streams of data simultaneously:
1.  **Visual Stream (Image)**: Analyzes the physical surface of the panel.
2.  **Time-Series Stream (Sensors)**: Analyzes historical voltage, current, and temperature data.

By fusing these inputs, our model predicts:
1.  **Defect Detection**: Identifies the specific type of fault (e.g., Bird-drop, Crack, Hotspot).
2.  **Defect Severity (%)**: Estimates how physically severe the damage is.
3.  **Future Health (%)**: Predicts the remaining performance capacity of the panel over the next 30 days.

---

## 🛠️ Tech Stack

*   **Language**: Python 3.x
*   **Deep Learning Framework**: PyTorch
*   **Visual Backbone**: VGG16 (Transfer Learning from ImageNet)
*   **Time-Series Backbone**: LSTM (Long Short-Term Memory Network)
*   **Data Processing**: NumPy, Pandas, Pillow (PIL)
*   **Simulation**: Custom Physics-based degradation simulator

---

## 🧠 System Methodology

### 1. Hybrid Model Architecture
Our architecture (`src/model.py`) is designed with a "Y" structure:

*   **Branch A (Visual)**: 
    *   Input: 224x224 RGB Image.
    *   Model: **VGG16** (Feature Extractor).
    *   Output: 25,088-dimensional feature vector representing texture and shape.
*   **Branch B (Time-Series)**:
    *   Input: 30-day sequence of [Voltage, Current, Temperature, Irradiance].
    *   Model: **2-Layer LSTM**.
    *   Output: 128-dimensional context vector capturing trends.
*   **Fusion Layer**:
    *   Concatenates Visual + Time-Series vectors.
    *   Passes through fully connected layers (Dense 4096 -> ReLU -> Dropout).

### 2. Multi-Head Predictions
The fused features feed into three separate prediction heads:
*   **Head 1: Classification** (Linear Layer). Outputs probabilities for 6 detected classes.
*   **Head 2: Severity Regression** (Sigmoid). Outputs 0.0 - 1.0 score.
*   **Head 3: Degradation Regression** (Sigmoid). Outputs 0.0 - 1.0 health score.

### 3. Logic: Generating Metrics
Since real-world labeled severity data effectively doesn't exist, we implemented a robust **Heuristic Logic** (`src/dataset.py`) to train the model:

#### A. Defect Severity Percentage
We map each visual class to a foundational severity score based on industry impact standards:
*   **Clean**: 0% Severity (Perfect)
*   **Dusty/Soiling**: 20% Severity (Easily fixable)
*   **Bird-drop**: 30% Severity (Localized heating risk)
*   **Snow-Covered**: 60% Severity (Major blockage)
*   **Electrical-damage**: 80% Severity (Internal failure)
*   **Physical-Damage**: 100% Severity (Permanent structural failure)

#### B. Predicted Health Percentage
This metric represents the "Remaining Useful Life" or performance efficiency. It is inversely derived from severity:
> **Health = 100% - (Severity * Impact Factor)**

For example, a panel with **Physical Damage (100% Severity)** might preserve only **20% Health**, whereas a **Dusty panel (20% Severity)** retains **84% Health**.

---

## 📂 Project Structure

```
Hybrid-W/
├── src/
│   ├── data/               # Consolidated dataset
│   ├── dataset.py          # Data loading & Severity synthesis logic
│   ├── model.py            # VGG16 + LSTM Architecture definition
│   ├── train.py            # Training loop (with multi-task loss)
│   ├── infer.py            # Inference script for single images
│   └── simulator.py        # Generates synthetic electrical data
├── requirements.txt        # Dependencies
└── hybrid_solar_model.pth  # Trained Model Weights
```

---

## 🚀 How to Run

### 1. Setup Environment
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
(Optional) If you want to improve accuracy or retrain from scratch:
```bash
python src/train.py
```
*Note: This runs quickly using our optimized 3-epoch demo configuration.*

### 3. Run Inference (Test a Panel)
To analyze a specific image and get a health report:
```bash
python src/infer.py "src/phy.jpg"
```

**Output Example:**
```
========================================
 HYBRID MODEL ANALYSIS REPORT
========================================
Image:              phy.jpg
----------------------------------------
Detected Condition: Physical-Damage
Classification Conf: 99.8%
----------------------------------------
Defect Severity:    99.5% (Estimated Extent)
Predicted Health:   11.3% (Performance after 30 days)
========================================
STATUS: MAINTENANCE REQUIRED (Physical-Damage)
URGENCY: HIGH - Defect Severity Critical
```

---
**Developed for the Wadla Hackathon**
