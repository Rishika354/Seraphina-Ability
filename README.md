# Seraphina: Depression Detection and Support Assistant

Seraphina is a machine learning-based application that predicts signs of depression using user input related to mental health, personal, and situational factors. The app provides personalized guidance and coping strategies depending on the severity level detected.

## 💡 Features

- Predicts depression levels: **No Depression**, **Moderate Depression**, or **Severe Depression**
- Trained using a Random Forest Classifier
- Based on real survey data
- Provides personalized mental health tips
- Supports input through Streamlit UI or command-line (CLI) script

## 🧠 Dataset

- File: `seraphina_dataset.csv`
- Cleaned and preprocessed with encoded categorical variables
- Target variable: `Depression_Rec1`
- Features include anxiety, personal circumstances, health status, and lifestyle

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit (for web interface, optional)
- CLI-compatible fallback script

## 🚀 How to Run

### 🖥️ Option 1: Streamlit Web App

```bash
pip install -r requirements.txt
streamlit run app.py
