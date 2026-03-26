"""
Script pour les expérimentations obligatoires du TP :
A. Distance euclidienne vs cosinus
B. Étude de l'effet du seuil (0.4, 0.6, 0.8)
C. Test sur mêmes personnes / personnes différentes

Mode d'emploi :
1) Placez vos images dans dataset/<personne>/...
2) Modifiez la liste image_pairs ci-dessous
3) Lancez : python experimentations.py
"""

from src.face_recognition_dl import FaceRecognitionDL


def print_results(title, results):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for row in results:
        print(
            f"Métrique={row['metric']:<10} | Seuil={row['threshold']:<4} | "
            f"Faux rejets={row['false_rejects']}/{row['total_same']} | "
            f"Fausses acceptations={row['false_accepts']}/{row['total_diff']}"
        )


def main():
    model = FaceRecognitionDL()
    model.build_database("dataset")

    # À adapter selon vos images réelles.
    image_pairs = [
        ("dataset/personne1/img1.jpg", "dataset/personne1/img2.jpg", 1),
        ("dataset/personne1/img1.jpg", "dataset/personne2/img1.jpg", 0),
    ]

    euclidean_thresholds = [0.4, 0.6, 0.8]
    cosine_thresholds = [0.4, 0.6, 0.8]

    euclidean_results = model.evaluate_thresholds(image_pairs, euclidean_thresholds, metric="euclidean")
    cosine_results = model.evaluate_thresholds(image_pairs, cosine_thresholds, metric="cosine")

    print_results("A/B/C - Expérimentation avec distance euclidienne", euclidean_results)
    print_results("A/B/C - Expérimentation avec similarité cosinus", cosine_results)


if __name__ == "__main__":
    main()
