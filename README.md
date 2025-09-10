# 🏦 Loan PDO (Post Due Obligation) Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pdo-predictor.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced machine learning system that predicts whether an approved loan will become a **Post Due Obligation (PDO)** or remain **Current (CUR)**. This project was developed during my internship at **NBK Egypt** to enhance loan risk assessment and improve banking decision-making processes.

---

## 🎯 Project Overview

The Loan PDO Predictor is a sophisticated risk assessment tool that analyzes customer profiles, loan characteristics, and financial metrics to predict loan performance. By identifying high-risk loans before they become problematic, this system helps financial institutions:

- **Reduce NPL ratios** by identifying potential defaults early
- **Optimize loan pricing** based on risk assessment
- **Improve portfolio management** through better risk segmentation
- **Enhance regulatory compliance** with Basel III requirements
- **Support data-driven lending decisions**

### 🚀 Live Demo
**[Access the Live Application](https://pdo-predictor.streamlit.app/)**

---

## 🔬 Problem Statement

Post Due Obligations (PDOs) represent loans that have missed payment schedules, potentially becoming Non-Performing Loans (NPLs). Early identification of such loans is crucial for:

- **Risk Mitigation**: Proactive measures to prevent defaults
- **Capital Allocation**: Better reserves planning and regulatory compliance
- **Customer Relationship**: Early intervention and restructuring opportunities
- **Profitability**: Reduced write-offs and improved portfolio performance

---

## 🧠 Model Architecture & Methodology

### Data Pipeline
```
Raw Customer Data → Feature Engineering → Encoding & Scaling → PCA → ML Model → Risk Prediction
```

### Key Components

1. **Feature Engineering Pipeline**
   - Automated calculation of 17+ derived financial ratios
   - Date-based feature extraction (year, month, day, days_since)
   - Categorical encoding with robust handling of unseen values

2. **Risk Metrics Calculation**
   - **DTI (Debt-to-Income) Ratio**: Total loan exposure relative to income
   - **Payment Burden**: Monthly payment as percentage of income
   - **Interest Coverage**: Loan sustainability metrics
   - **Exposure Ratios**: Risk concentration analysis

3. **Model Pipeline**
   - **Preprocessing**: StandardScaler for numerical features
   - **Dimensionality Reduction**: PCA (95% variance retention)
   - **Class Balancing**: SMOTE for handling imbalanced datasets
   - **Classification**: Random Forest with class weights optimization

4. **Prediction Output**
   - **Binary Classification**: PDO (0) vs CUR (1)
   - **Probability Scores**: Risk assessment with confidence levels
   - **Feature Importance**: Explainable AI for decision transparency

---

## ✨ Key Features

### 🎛️ **Dual Input Modes**
- **Human-Friendly Interface**: Organized forms with intuitive sections
- **CSV Batch Processing**: Quick predictions for multiple applications

### 📊 **Advanced Risk Analytics**
- Real-time calculation of 17+ financial ratios
- Interactive risk metrics dashboard
- Automated loan payment calculations
- Comprehensive exposure analysis

### 🔍 **Explainable AI**
- Feature vector visualization
- Model prediction probabilities
- Technical processing details
- Risk factor breakdown

### 🎨 **Professional UI/UX**
- Modern, responsive design
- Gradient-based visual hierarchy
- Enhanced user experience with custom CSS
- Mobile-friendly interface

---

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

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/loan-predictor-streamlit.git
   cd loan-predictor-streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the model** (if you have training data)
   ```bash
   python Model.py
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the interface**
   Open your browser and navigate to `http://localhost:8501`

---

## 📋 Usage Guide

### Method 1: Human-Friendly Input
1. **Geographical Information**: Customer location and residency details
2. **Personal & Demographics**: Age, income, employment details
3. **Account Information**: Banking relationship and KYC details
4. **Loan Details**: Amount, duration, interest rate, and type
5. **Review & Predict**: Examine calculated metrics and get prediction

### Method 2: CSV Batch Processing
1. Prepare CSV with customer data
2. Paste the row into the text area
3. Click "Predict from CSV" for instant results

---

## 📊 Model Performance

The model has been trained and validated on real banking data with the following characteristics:

- **Dataset**: Customer loan applications with historical performance
- **Features**: 50+ engineered features including financial ratios and risk metrics
- **Target**: Binary classification (PDO vs CUR)
- **Validation**: Stratified cross-validation with ROC-AUC optimization
- **Deployment**: Production-ready with robust error handling

---

## 🏗️ Project Structure

```
loan-predictor-streamlit/
├── app.py                  # Main Streamlit application
├── Model.py               # Model training and artifact generation
├── style.css              # Custom CSS styling
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── model.pkl             # Trained ML model
├── scaler.pkl            # Feature scaler
├── pca.pkl               # PCA transformer
├── features.pkl          # Feature list
├── defaults.pkl          # Default values
├── encoders.pkl          # Categorical encoders
└── metadata.pkl          # Feature metadata
```

---

## 🔧 Technical Implementation

### Feature Engineering
- **Derived Metrics**: 17 automatically calculated financial ratios
- **Date Features**: Multi-dimensional date encoding
- **Risk Indicators**: DTI, payment burden, exposure ratios

### Model Training
- **Data Preprocessing**: Robust handling of missing values and outliers
- **Feature Selection**: PCA for optimal dimensionality
- **Class Balancing**: SMOTE for addressing dataset imbalance
- **Model Selection**: Random Forest with hyperparameter tuning

### Production Deployment
- **Artifact Management**: Serialized models and preprocessors
- **Error Handling**: Comprehensive exception management
- **Performance**: Optimized for real-time predictions
- **Scalability**: Cloud-ready architecture

---

## 🎓 Professional Impact

This project was developed during my internship at **NBK Egypt**, one of the leading banks in the Middle East and Africa. The system demonstrates:

- **Real-world Application**: Addresses actual banking challenges
- **Industry Standards**: Follows best practices in financial modeling
- **Technical Excellence**: Production-quality code and architecture
- **Business Value**: Quantifiable impact on risk management

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Ali Tharwat** - alimotharwat@gmail.com - [LinkedIn](https://linkedin.com/in/ali-tharwat)

Project Link: [https://github.com/ali-tharwat/loan-predictor-streamlit](https://github.com/ali-tharwat/loan-predictor-streamlit)
