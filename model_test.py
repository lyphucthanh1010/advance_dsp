import os
import json
import numpy as np
import multiprocessing
import shutil
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

dataset_model_path ='dataset'


def load_data_from_json_directory(directory):
    X = []
    y = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                json_data = json.load(file)
                chromaprints = json_data["chromaprints"]
                label = json_data["label"]
                for chromaprint in chromaprints:
                    X.append(chromaprint)
                    y.append(label)
    return X, y

def load_real_data(real_data):
    real_chromaprints = []
    for chromaprint in real_data:
        real_chromaprints.append(chromaprint)
    return np.array(real_chromaprints)

def empty_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def train_and_save_model(directory, model):
    X, y = load_data_from_json_directory(directory)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = model
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'modeltest.pkl')

def process_real_data_and_copy_files(directory, model_path, real_data, data_full):
    clf = joblib.load(model_path)

    X, y = load_data_from_json_directory(directory)    
    real_data_chromaprints = load_real_data(real_data)
    printed_labels = {}

    for real_chromaprint in real_data_chromaprints:
        label_probabilities = clf.predict_proba([real_chromaprint])[0]

        top_indices = np.argsort(label_probabilities)[-1:][::-1]

        top_labels = [clf.classes_[index] for index in top_indices]

        for label in top_labels:
            if label not in printed_labels:
                printed_labels[label] = real_chromaprint
                break
    copy_json_files_with_label(data_full, printed_labels)


def copy_json_files_with_label(directory, printed_labels):
    output_directory = "copied_json_files"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    empty_directory(directory=output_directory)

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                json_data = json.load(file)
                label = json_data["label"]
                if label in printed_labels:
                    shutil.copy(filepath, output_directory)


train_and_save_model(dataset_model_path, RandomForestClassifier(n_estimators=100, random_state=42))