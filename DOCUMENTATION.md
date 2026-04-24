# Technical Documentation: AB-XAI-RNN-SGRU Model

This document provides in-depth technical details about the model architecture, feature engineering, and the XAI implementation used in this fraud detection system.

## 🧠 Model Architecture

The **AB-XAI-RNN-SGRU** model is designed for high-accuracy fraud detection in imbalanced financial datasets.

### 1. Spatial Feature Extraction (1D-CNN)
Even though the data is tabular, we treat it as a 1D sequence or a vector of local patterns. The 1D-CNN layer (filters=64, kernel=1) identifies local correlations between features.

### 2. Sequential/Temporal Modeling (SGRU)
A **Stacked Gated Recurrent Unit (SGRU)** is used for sequential modeling. 
-   **First GRU**: 128 units, `return_sequences=True`.
-   **Second GRU**: 64 units, `return_sequences=True`.
-   This architecture is effective for capturing long-term dependencies if the transaction history is provided as a sequence.

### 3. Attention Mechanism (Custom Layer)
The Attention Layer calculates a context vector by weighting the GRU outputs.
-   **Weights ($W$ and $b$)**: Learnable parameters used to score each feature's importance.
-   **Softmax**: Normalizes the importance scores (alphas).
-   **Context Vector**: Weighted sum of features, emphasizing the most critical data points for the final decision.

### 4. Output Layer
A Dense layer with 32 units (`relu`) followed by a single unit Sigmoid output for binary classification.

## 📊 XAI Strategy

The system uses a "Robust Tri-Layer" XAI approach:

### 1. Attention-Based XAI (Built-in)
The model itself provides attention weights (`alphas`) for every inference. This is the fastest XAI layer but can sometimes be "noisier" than external explainers.

### 2. SHAP (KernelExplainer)
SHAP provides a mathematically rigorous attribution for each feature.
-   **Pros**: Game-theory based, robust.
-   **Implementation**: Uses a background dataset (training sample) to calculate baseline expectations.

### 3. LIME (Local Surrogate)
LIME builds a local linear model around a specific instance to explain why it was classified in a certain way.
-   **Pros**: Excellent for understanding "edge case" transactions.
-   **Robustness**: If PCA or other transformations are used, LIME operates on the *processed* feature space, and the system maps these back to meaningful "Component" labels if necessary.

## 🛡️ Preprocessing (EPCA & SMOTE)

-   **EPCA (Enhanced PCA)**: Reduces noise and dimensionality by focusing on components explaining 95% of variance.
-   **SMOTE**: Solves the "class imbalance" problem (where legit > fraud 99:1) by synthetically generating minority class samples during training.

## 🔗 Implementation Details

-   **Backend**: Flask 3.1+ with CORS support.
-   **Deep Learning**: TensorFlow 2.x / Keras 3.x.
-   **Custom Logic**: The `AB_XAI_RNN_Model` in `model/ab_xai_rnn.py` handles the dual-output (prob + alpha) required for synchronous prediction and explanation.
