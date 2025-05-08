from module import predict_image
import pickle

def main():
    # Load model yang sudah dilatih
    with open("model/model_svm.pkl", "rb") as f:
        model = pickle.load(f)

    # Uji gambar
    image_path = "gambar_uji.jpg"  # ubah ke gambar yang mau diuji
    predicted_label = predict_image(model, image_path)
    print(f"Prediksi: {predicted_label}")

if __name__ == "__main__":
    main()