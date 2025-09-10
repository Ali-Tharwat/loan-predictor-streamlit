# Loan Repayment Risk Prediction

This project is a Streamlit web application that predicts the risk of a customer defaulting on a loan. It uses a pre-trained machine learning model to make predictions based on customer data.

## 🚀 Live Demo

You can access the live application here: **[https://pdo-predictor.streamlit.app/](https://pdo-predictor.streamlit.app/)**

-----

## Features

  * **User-Friendly Interface**: An intuitive web interface for entering customer data.
  * **Two Input Modes**:
      * **Human-friendly**: A form with organized sections for easy data entry.
      * **CSV Paste**: Paste a row of comma-separated values for quick predictions.
  * **Real-time Predictions**: Get instant loan repayment risk predictions.
  * **Automated Feature Engineering**: Automatically calculates derived financial ratios and loan values from the input data.
  * **Explainable AI (XAI)**: Shows the transformed feature vector and model prediction probabilities to provide insight into the prediction.

-----

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

* **Python**: The core programming language used for the application.
* **Pandas**: Used for data manipulation and analysis, particularly for handling the input data.
* **NumPy**: A fundamental package for numerical computations, used for handling arrays and mathematical operations.
* **Scikit-learn**: The primary machine learning library used for building the model pipeline, including preprocessing, dimensionality reduction (PCA), and the final classification model.
* **Streamlit**: The framework used to create the interactive web application interface.
-----

## How to Run Locally

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/loan-predictor-streamlit.git
    cd loan-predictor-streamlit
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

4.  Open your web browser and go to `http://localhost:8501`.

-----

## File Descriptions

  * `Model.py`: The model building Python script.
  * `app.py`: The main Streamlit application file.
  * `model.pkl`: The pre-trained machine learning model.
  * `scaler.pkl`: The scaler used to standardize the data.
  * `pca.pkl`: The PCA model for dimensionality reduction.
  * `features.pkl`: A list of the features used by the model.
  * `defaults.pkl`: A dictionary of default values for the input fields.
  * `encoders.pkl`: A dictionary of label encoders for categorical features.
  * `metadata.pkl`: A dictionary of metadata about the features.
  * `requirements.txt`: A list of the Python dependencies.

-----

## Model Architecture

The prediction model is a machine learning pipeline that consists of the following steps:

1.  **Data Preprocessing**:
      * Categorical features are encoded using label encoders.
      * Numerical features are scaled using a standard scaler.
2.  **Dimensionality Reduction**:
      * Principal Component Analysis (PCA) is used to reduce the number of features.
3.  **Prediction**:
      * A pre-trained classification model is used to predict the probability of loan default and classify the result as "Approved" or "Rejected".
