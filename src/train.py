import cv2
import os
import numpy as np

def train_model():
    print("\n --- Training Model ......---")
    dataset_path = 'faces_dataset'
    config_path = 'config'

    if not os.path.exists(config_path):
        os.makedirs(config_path)

    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path))==0:
        print(" Error: No student data found.")
        return 

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    # Loop through dataset
    print("Processing images......")
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".jpg"):
            # Extract label (student name) from filename
            label = file_name.split("_")[0]

            img_path = os.path.join(dataset_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            faces.append(img)
            labels.append(label)

    # Convert labels to integers (map names → IDs)
    label_ids = {name: idx for idx, name in enumerate(set(labels))}
    y = [label_ids[label] for label in labels]

    recognizer.train(faces, np.array(y))

    # Save model inside config directory
    model_file = os.path.join(config_path, "trained_model.yml")
    recognizer.save(model_file)

    print("Model trained and saved as trained_model.yml")
    print("Label mapping:", label_ids)
    return label_ids
