import os
import pickle
from module import load_dataset, classification

def save_model(model, filename="model/model_svm.pkl"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model disimpan di: {filename}")

def main():
    dataset_dir = "dataset"
    data_features, labels = load_dataset(dataset_dir)

    if not data_features:
        print("Dataset kosong atau gagal diproses.")
        return

    model = classification(data_features, labels)
    save_model(model)

if __name__ == "__main__":
    main()