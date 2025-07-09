import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def display_no_depression_tips():
    print("\n🟢 No Depression Detected")
    print("It seems that no signs of depression were detected.")
    print("Tips to maintain mental well-being:")
    print("- Practice mindfulness and relaxation techniques.")
    print("- Engage in regular physical activity.")
    print("- Stay connected with loved ones.")
    print("- Seek professional help if needed.")

def display_moderate_depression_tips():
    print("\n🟡 Moderate Depression Detected")
    print("It appears that signs of moderate depression were detected.")
    print("Tips to help cope with depression:")
    print("- 🌈 Radiant Resilience: Embrace your inner superhero!")
    print("- 🎨 Bold Self-Care Brushstrokes: Indulge in self-care.")
    print("- 💬 Vibrant Venting Sessions: Let your feelings breathe.")
    print("- 🌞 Sunshine Seeking: Get outdoors, even briefly.")

def display_severe_depression_tips():
    print("\n🔴 Severe Depression Detected")
    print("It seems that signs of severe depression were detected.")
    print("Important steps to take:")
    print("- Contact a mental health professional or therapist.")
    print("- Reach out to a trusted friend or family member.")
    print("- Consider helplines or support groups.")

# Load the dataset
f = "ML-DataSet_5.csv"
df = pd.read_csv(f)
df1 = df.drop(['f_id', 'duplicate_x', 'duplicate_y', 'duplicate_z', 'duplicate_v', 'duplicate_w', 
               'duplicate_a', 'dup_b', 'dup_c', 'Anxeity_Rec'], axis=1)
df1.fillna(df1.mean(), inplace=True)

# Split data into features and labels
X = df1.drop(['Depression_Rec1'], axis=1)
y = df1['Depression_Rec1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

def predict_depression_rec(input_data):
    input_data_np = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_np)
    prediction = rf_classifier.predict(std_data)
    return prediction[0]

def main():
    print("🔍 Seraphina - Depression Prediction Tool")
    print("Enter your values below (using expected numerical encodings).")

    # Sample inputs (replace with actual user input or integrate with other input method)
    input_data = [
        1, 1, 1, 1, 1,    # first 5 features
        2, 2, 2, 2, 2,    # next 5 features
        2, 2, 2, 2, 2,    # next 5 features
        2, 2, 2, 2, 2,    # next 5 features
        2, 2, 2, 1, 1,    # next 5 features
        1, 1, 1, 2, 1,    # next 5 features
        1, 1, 1, 1, 1,    # next 5 features
        1, 1              # last 2 features
    ]

    predicted_outcome = predict_depression_rec(input_data)

    if predicted_outcome == 1:
        display_moderate_depression_tips()
    elif predicted_outcome == 2:
        display_no_depression_tips()
    elif predicted_outcome == 3:
        display_severe_depression_tips()
    else:
        print("⚠️ Unable to determine result.")

if __name__ == '__main__':
    main()
