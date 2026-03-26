from src.face_recognition_dl import FaceRecognitionDL


def main():
    dataset = "dataset"
    test_image = "test_images/test.jpg"

    model = FaceRecognitionDL(similarity_metric="euclidean")

    print("Construction de la base d'embeddings...")
    model.build_database(dataset)

    print("Reconnaissance en cours...")
    label, distance, decision = model.recognize(test_image, threshold=0.8, metric="euclidean")

    print("\nRésultat :")
    print("Identité :", label)
    print("Distance :", round(distance, 4))
    print("Décision :", decision)


if __name__ == "__main__":
    main()
