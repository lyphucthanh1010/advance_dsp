import os
import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

datafull_path = "datasetreal"
model_path = "model.pkl"

def evaluate_model_real():
    clf = joblib.load(model_path)
    
    X = []
    y_true = []
    
    for filename in os.listdir(datafull_path):
        if filename.endswith(".json"):
            filepath = os.path.join(datafull_path, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                chromaprints = data.get("chromaprints", [])
                label = data.get("label", "")
                X.append(chromaprints)
                y_true.append(label)
    
    X = np.array(X)
    y_true = np.array(y_true)
    
    print("Shape of X:", X.shape)  
    if X.shape[1] != 30:
        print("Transforming data from {} features to 30 features.".format(X.shape[1]))
        X = X[:, :30]
    
    y_pred = clf.predict(X)
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print("Accuracy:", acc)
    # print("Classification Report:")
    # print(report)
    
    return acc, report

if __name__ == "__main__":
    evaluate_model_real()
