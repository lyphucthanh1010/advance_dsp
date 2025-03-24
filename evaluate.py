import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
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
    report = classification_report(y_true, y_pred, output_dict=True)
    
    print("Accuracy:", acc)
    print("Classification Report:")
    print(report)
    
    return acc, report

def plot_accuracy_and_report(acc, report):
    # Plot overall accuracy
    plt.figure(figsize=(5,5))
    plt.bar(["Accuracy"], [acc], color="skyblue")
    plt.ylim([0, 1])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy Score")
    plt.savefig("accuracy.png")  # Save the accuracy plot to a file
    plt.close()  # Close the figure

    # Prepare classification report data for plotting
    classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [report[cls][metric] for cls in classes] for metric in metrics}
    
    x = np.arange(len(classes))
    width = 0.2
    plt.figure(figsize=(10,6))
    plt.bar(x - width, data['precision'], width, label='Precision')
    plt.bar(x, data['recall'], width, label='Recall')
    plt.bar(x + width, data['f1-score'], width, label='F1-Score')
    
    plt.xticks(x, classes, rotation=90)
    plt.ylim([0, 1])
    plt.title("Classification Report Metrics")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("classification_report.png")  # Save the classification report plot to a file
    plt.close()


if __name__ == "__main__":
    accuracy, report = evaluate_model_real()
    plot_accuracy_and_report(accuracy, report)
