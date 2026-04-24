# Premium Fraud Detection System with AB-XAI-RNN-SGRU

A state-of-the-art financial cybersecurity framework for real-time fraud detection, featuring Explainable AI (XAI) for transparency and trust.

## 🚀 Overview

This project implements the **AB-XAI-RNN-SGRU** model as described in modern financial security research. It combines deep learning with robust preprocessing and multiple XAI layers (SHAP, LIME, and Attention) to provide not just a "Fraud/Legit" decision, but a clear explanation of *why* a transaction was flagged.

## 🏗️ Architecture

The system follows a multi-stage pipeline:

1.  **Preprocessing Module**:
    -   **Categorical Encoding**: Handles string labels and IDs.
    -   **Scaling**: Standardizes numerical features.
    -   **PCA (EPCA)**: Optional dimensionality reduction for high-dimensional datasets.
    -   **SMOTE**: Rebalances imbalanced datasets during training.

2.  **Model Core (AB-XAI-RNN-SGRU)**:
    -   **1D-CNN**: Extracts spatial/local patterns from tabular data.
    -   **Stacked GRU (SGRU)**: Captures temporal or sequential dependencies (even for single-event tabular data treated as 1-step sequences).
    -   **Attention Layer**: Calculates feature importance weights dynamically during inference.

3.  **XAI Layer**:
    -   **Attention Weights**: Real-time "internal" model focus.
    -   **SHAP**: Global/Local feature contribution analysis.
    -   **LIME**: Local surrogate interpretation for specific instances.

## 📁 Project Structure

-   `app.py`: Flask backend serving the API.
-   `model/`: Core logic package.
    -   `ab_xai_rnn.py`: Model definition and training/inference logic.
    -   `attention.py`: Custom Keras AttentionLayer implementation.
    -   `preprocessing.py`: Data cleaning and transformation pipeline.
    -   `xai_explainer.py`: Integration of SHAP and LIME.
-   `static/`: Frontend assets (UI/UX).
-   `templates/`: HTML templates.
-   `test_endpoints.py`: Automated API testing suite.

## 🛠️ Setup & Installation

1.  **Prerequisites**: Python 3.10+
2.  **Installation**:
    ```bash
    py -m pip install -r requirements.txt
    ```

## 🚥 API Endpoints

-   `POST /api/upload`: Upload a CSV dataset (BankSim, CreditCard, etc.).
-   `POST /api/predict_batch`: Run batch analysis on an uploaded file.
-   `POST /api/predict_single`: Predict and explain a manual entry.
-   `POST /api/train`: Re-train the model on a specific dataset.
-   `GET /api/live_stream`: Simulate real-time transaction monitoring.

## 🖥️ Usage

### Running the Server
```bash
py app.py
```

### Running Tests
```bash
py test_endpoints.py
```

## 💎 Design Philosophy

The UI is built with a **Glassmorphism** aesthetic, emphasizing:
-   **Visual Clarity**: Real-time charts and gauge meters for risk.
-   **Explainability**: Interactive breakdown of feature importance for every flagged transaction.
-   **Responsiveness**: Optimized for high-frequency monitoring.
