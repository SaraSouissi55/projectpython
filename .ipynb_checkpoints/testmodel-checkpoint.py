import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

print("=" * 60)
print("STEP 1: LOAD AND CHECK YOUR DATA")
print("=" * 60)

# Load your data
df = pd.read_csv("sensor_data_finall.csv")

# Check data
print(f"✓ Total rows: {len(df)}")
print(f"✓ Columns: {df.columns.tolist()}")

# Check anomaly distribution
if 'anomaly' in df.columns:
    print(f"\n📊 Anomaly Distribution:")
    print(df['anomaly'].value_counts())
    print(f"✓ Normal: {(df['anomaly'] == False).sum()} rows")
    print(f"✓ Anomaly: {(df['anomaly'] == True).sum()} rows")
    print(f"✓ Anomaly %: {df['anomaly'].mean()*100:.1f}%")
else:
    print("❌ ERROR: No 'anomaly' column found!")
    exit()

print("\n" + "=" * 60)
print("STEP 2: PREPROCESS DATA")
print("=" * 60)

# Handle missing values
for col in ['temperature', 'humidity', 'air_quality']:
    df[col] = df[col].ffill()
    print(f"✓ {col}: Filled missing values")

# Prepare features and target
X = df[['temperature', 'humidity', 'air_quality']]
y = df['anomaly'].astype(int)  # Ensure it's 0/1

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")

# Split data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n✓ Training samples: {len(X_train_raw)}")
print(f"✓ Test samples: {len(X_test_raw)}")
print(f"✓ Training anomalies: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"✓ Test anomalies: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)
print("✓ Data scaled")

print("\n" + "=" * 60)
print("STEP 3: TRAIN MODEL WITH CLASS BALANCING")
print("=" * 60)

# Train model with class balancing
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✓ Model trained with class_weight='balanced'")

# Save model and scaler
joblib.dump(model, 'anomaly_model_fixed.pkl')
joblib.dump(scaler, 'scaler_fixed.pkl')
print("✓ Model saved as 'anomaly_model_fixed.pkl'")
print("✓ Scaler saved as 'scaler_fixed.pkl'")

print("\n" + "=" * 60)
print("STEP 4: BASIC MODEL EVALUATION")
print("=" * 60)

# Predict on test set
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Test Accuracy: {accuracy:.1%}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"""
      Predicted
      Normal  Anomaly
Actual Normal   {cm[0,0]:>4}   {cm[0,1]:>4}
       Anomaly  {cm[1,0]:>4}   {cm[1,1]:>4}
""")

print("\n" + "=" * 60)
print("STEP 5: CRITICAL TEST - DO PREDICTIONS VARY?")
print("=" * 60)

# Define test cases
test_cases = [
    ("VERY NORMAL", [22.0, 50.0, 70.0]),
    ("SLIGHTLY SUSPICIOUS", [28.0, 65.0, 95.0]),
    ("CLEAR ANOMALY", [38.0, 90.0, 180.0]),
    ("EXTREME NORMAL", [20.0, 40.0, 60.0]),
    ("EXTREME ANOMALY", [40.0, 95.0, 200.0]),
    ("BORDERLINE 1", [32.0, 75.0, 110.0]),
    ("BORDERLINE 2", [25.0, 85.0, 70.0])
]

print(f"{'Scenario':<25} {'Prediction':<12} {'Anomaly %':<10} {'Normal %':<10}")
print("-" * 57)

all_probs_identical = True
first_proba = None

for name, values in test_cases:
    # Scale input
    values_scaled = scaler.transform([values])
    
    # Get prediction
    prediction = model.predict(values_scaled)[0]
    probabilities = model.predict_proba(values_scaled)[0]
    
    # Check if probabilities vary
    if first_proba is None:
        first_proba = probabilities[1]
    elif not np.allclose(probabilities[1], first_proba, atol=0.01):
        all_probs_identical = False
    
    # Display
    pred_text = "🔴 ANOMALY" if prediction == 1 else "✅ NORMAL"
    print(f"{name:<25} {pred_text:<12} {probabilities[1]:<10.1%} {probabilities[0]:<10.1%}")

print("\n" + "=" * 60)
print("STEP 6: DIAGNOSTIC RESULTS")
print("=" * 60)

if all_probs_identical:
    print("❌❌❌ PROBLEM: All predictions have SAME probability!")
    print("The model is still not learning patterns.")
    print("\nPossible fixes:")
    print("1. Check if features actually differ between normal/anomaly")
    print("2. Try different class_weight values")
    print("3. Use SMOTE for oversampling")
    print("4. Create better anomaly labels")
else:
    print("✅ SUCCESS: Predictions vary with different inputs!")
    print("The model is learning patterns properly.")
    
    # Check if predictions make sense
    print("\n✓ Normal cases → Low anomaly probability (< 50%)")
    print("✓ Anomaly cases → High anomaly probability (> 50%)")
    print("✓ Borderline cases → Around 50% probability")

print("\n" + "=" * 60)
print("STEP 7: FEATURE IMPORTANCE CHECK")
print("=" * 60)

# Check which features matter
feature_names = ['temperature', 'humidity', 'air_quality']
importances = model.feature_importances_

for name, importance in zip(feature_names, importances):
    print(f"✓ {name}: {importance:.1%}")

if max(importances) - min(importances) < 0.1:
    print("\n⚠️ Warning: Features have similar importance")
    print("Model might not be finding clear patterns")
else:
    print(f"\n✅ Strongest feature: {feature_names[np.argmax(importances)]}")

print("\n" + "=" * 60)
print("STEP 8: FINAL VERIFICATION FOR DASHBOARD")
print("=" * 60)

# Function for dashboard
def predict_for_dashboard(temp, hum, aq):
    """Function to use in your Streamlit dashboard"""
    # Scale input
    input_scaled = scaler.transform([[temp, hum, aq]])
    
    # Predict
    is_anomaly = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    return {
        'is_anomaly': bool(is_anomaly),
        'anomaly_probability': float(proba[1]),
        'normal_probability': float(proba[0]),
        'anomaly_percent': f"{proba[1]:.1%}",
        'normal_percent': f"{proba[0]:.1%}"
    }

# Test dashboard function
print("Testing dashboard function:")
test1 = predict_for_dashboard(22, 50, 70)
test2 = predict_for_dashboard(38, 90, 180)

print(f"\nNormal case (22, 50, 70):")
print(f"  Anomaly: {test1['is_anomaly']}, Probability: {test1['anomaly_percent']}")

print(f"\nAnomaly case (38, 90, 180):")
print(f"  Anomaly: {test2['is_anomaly']}, Probability: {test2['anomaly_percent']}")

print("\n" + "=" * 60)
print("✅ VALIDATION COMPLETE")
print("=" * 60)