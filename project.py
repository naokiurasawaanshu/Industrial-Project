import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier 
from xgboost import XGBClassifier 
import joblib 
 
# Load dataset 
df = pd.read_csv('data/sensor_readings.csv') 
 
# Preprocess 
X = df.drop(['timestamp', 'machine_failure'], axis=1) 
y = df['machine_failure'] 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
 
models = { 
"Logistic Regression": LogisticRegression(), 
"Random Forest": RandomForestClassifier(n_estimators=100), 
"XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 
"SVM": SVC(), 
"MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300) 
} 
 
for name, model in models.items(): 
model.fit(X_train_scaled, y_train) 
preds = model.predict(X_test_scaled) 
print(f"\n{name} Results:") 
print(classification_report(y_test, preds)) 
joblib.dump(model, f"models/model_{name.lower().replace(' ', '_')}.pkl")

import joblib 
import pandas as pd 
 
model = joblib.load("models/model_random_forest.pkl") 
scaler = joblib.load("models/scaler.pkl") 
 
# Simulated input 
sample = pd.DataFrame([{
  "temperature": 90,
 "vibration": 0.04,
 "pressure": 31,
 "heating_level": 65
}]) 
 
sample_scaled = scaler.transform(sample) 
result = model.predict(sample_scaled) 
print(" fine.")
