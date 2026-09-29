import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class DelayPredictor:
    def __init__(self):
        # Using Random Forest to capture nonlinear interactions between weather, traffic, backlog
        self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.is_trained = False
        
    def train(self, df: pd.DataFrame):
        """Trains the ML model on completed trip data."""
        if len(df) < 100:
            print("Not enough data to train ML predictor. Need at least 100 samples.")
            return False
            
        # Features: avg_weather, avg_traffic, avg_backlog (expected_eta dropped to prevent leakage)
        X = df[['avg_weather', 'avg_traffic', 'avg_backlog']]
        y = df['is_delayed']
        
        # Add some noise if all labels are the same to prevent crash during PoC
        if len(y.unique()) == 1:
            print("Not enough class variance to train.")
            return False
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        from sklearn.metrics import classification_report
        report = classification_report(y_test, preds, zero_division=0)
        
        self.is_trained = True
        print(f"Random Forest trained. Validation Accuracy: {acc:.2f}")
        print("Classification Report:\n", report)
        return True
        
    def predict_delay_probability(self, avg_weather: float, avg_traffic: float, avg_backlog: float, expected_eta: float) -> float:
        """Predicts probability of delay for a given scenario."""
        if not self.is_trained:
            # Fallback if not trained: simple heuristic
            risk = (avg_weather * 0.4) + (avg_traffic * 0.4) + (avg_backlog * 0.2)
            return min(1.0, risk)
            
        X_new = pd.DataFrame([{
            'avg_weather': avg_weather, 
            'avg_traffic': avg_traffic, 
            'avg_backlog': avg_backlog
        }])
        
        probs = self.model.predict_proba(X_new)[0]
        # Assuming class 1 is "delayed"
        if len(self.model.classes_) == 2:
            delayed_index = list(self.model.classes_).index(1)
            return probs[delayed_index]
        return 0.0

if __name__ == "__main__":
    predictor = DelayPredictor()
    # Dummy data test
    df = pd.DataFrame({
        'avg_weather': np.random.rand(100),
        'avg_traffic': np.random.rand(100),
        'avg_backlog': np.random.rand(100),
        'expected_eta': np.random.uniform(5, 20, 100),
        'is_delayed': np.random.randint(0, 2, 100)
    })
    predictor.train(df)
    prob = predictor.predict_delay_probability(0.8, 0.9, 0.5, 10.0)
    print(f"Delay Probability: {prob:.2f}")
