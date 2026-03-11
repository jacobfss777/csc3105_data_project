import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Prepare Features (X) and Target (y)
# We drop the features that showed zero importance in the previous step
features_to_keep = [
    'RXPACC', 'CIR_PWR', 'FP_AMP3', 'FP_AMP2', 'MAX_NOISE', 
    'FP_AMP1', 'STDEV_NOISE', 'FP_IDX', 'Measured range (time of flight)'
]

X = df_clean[features_to_keep]
y = df_clean['NLOS']

# 2. Data Split (Rubric requirement: Decide the split ratio) 
# We'll use an 80:20 split as it's standard for a dataset of 42,000 samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 3. Model Training [cite: 209]
# We'll use Random Forest as our first classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluation [cite: 212]
y_pred = clf.predict(X_test)

print("\n--- Model Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Visualization: Confusion Matrix [cite: 214]
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted LOS', 'Predicted NLOS'],
            yticklabels=['Actual LOS', 'Actual NLOS'])
plt.title('Confusion Matrix: UWB Signal Classification')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()