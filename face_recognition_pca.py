#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class FaceRecognitionPCA:
    """
    Reconnaissance faciale par PCA (Eigenfaces) + détection Viola-Jones.
    """

    def __init__(self, n_components: int = 30, face_size: Tuple[int, int] = (100, 100)):
        """
        Initialise :
        - détecteur Viola-Jones
        - nombre de composantes principales
        - variables internes (mean, eigenvectors, projections, labels)
        """
        self.n_components = int(n_components)
        self.face_size = face_size
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if self.detector.empty():
            raise RuntimeError("Impossible de charger la cascade de Haar OpenCV.")

        self.mean: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.projections: Optional[np.ndarray] = None
        self.labels: List[str] = []
        self.training_faces: Optional[np.ndarray] = None
        self.image_paths: List[str] = []

    def detect_face(self, image: np.ndarray, return_bbox: bool = False):
        """
        Entrée : image (BGR)
        Sortie : visage détecté en niveaux de gris (100x100)
        Étapes :
        - conversion en gris
        - detectMultiScale
        - sélection du plus grand visage
        - resize 100x100
        """
        if image is None:
            return (None, None) if return_bbox else None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )

        if len(faces) == 0:
            return (None, None) if return_bbox else None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, self.face_size, interpolation=cv2.INTER_AREA)

        if return_bbox:
            return face, (int(x), int(y), int(w), int(h))
        return face

    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parcourt un dossier structuré par personne :
            dataset/
              person1/
              person2/
        Pour chaque image :
        - détecter le visage
        - vectoriser
        - stocker dans X
        - enregistrer le label
        Retour : X (numpy array), y (labels)
        """
        dataset = Path(dataset_path)
        if not dataset.exists() or not dataset.is_dir():
            raise FileNotFoundError(f"Dossier introuvable : {dataset_path}")

        X = []
        y = []
        image_paths = []
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for person_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
            label = person_dir.name
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in valid_ext:
                    continue
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                face = self.detect_face(image)
                if face is None:
                    continue
                X.append(face.flatten().astype(np.float64))
                y.append(label)
                image_paths.append(str(img_path))

        if not X:
            raise ValueError(
                "Aucun visage détecté dans le dataset. Vérifiez les images et la structure des dossiers."
            )

        self.labels = y
        self.image_paths = image_paths
        self.training_faces = np.array(X, dtype=np.float64)
        return self.training_faces, np.array(y)

    def compute_pca(self, X: np.ndarray) -> None:
        """
        Étapes :
        - calcul moyenne
        - centrage
        - covariance (forme réduite)
        - valeurs propres
        - tri décroissant
        - sélection n_components
        Stocker :
            self.mean
            self.eigenvectors
        """
        if X.ndim != 2:
            raise ValueError("X doit être une matrice 2D (n_samples, n_features).")

        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("Il faut au moins 2 images pour calculer la PCA.")

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        reduced_cov = np.dot(X_centered, X_centered.T) / (n_samples - 1)
        eigenvalues, eigenvectors_small = np.linalg.eigh(reduced_cov)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors_small = eigenvectors_small[:, order]

        k = min(self.n_components, n_samples)
        eigenfaces = []
        for i in range(k):
            v = np.dot(X_centered.T, eigenvectors_small[:, i])
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                eigenfaces.append(v / norm)

        if not eigenfaces:
            raise ValueError("Échec du calcul des eigenfaces.")

        self.eigenvectors = np.column_stack(eigenfaces)
        self.projections = np.dot(X_centered, self.eigenvectors)

    def project(self, face_vector: np.ndarray) -> np.ndarray:
        """
        Projection d'un visage dans l'espace PCA.
        """
        if self.mean is None or self.eigenvectors is None:
            raise RuntimeError("Le modèle PCA n'a pas encore été appris.")

        face_vector = face_vector.astype(np.float64).flatten()
        centered = face_vector - self.mean
        return np.dot(centered, self.eigenvectors)

    def train(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge la base d'apprentissage et construit le modèle PCA.
        """
        X, y = self.load_dataset(dataset_path)
        self.compute_pca(X)
        return X, y

    def recognize(self, image_path: str, threshold: float = 2500.0) -> Dict[str, object]:
        """
        Retourne :
        - identité la plus proche
        - distance minimale
        - décision (Match / No Match)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Impossible de lire l'image test : {image_path}")

        face, bbox = self.detect_face(image, return_bbox=True)
        if face is None:
            return {
                "identity": "Aucun visage détecté",
                "distance": float("inf"),
                "decision": "No Match",
                "bbox": None,
                "image": image,
                "face": None,
            }

        face_vector = face.flatten().astype(np.float64)
        test_projection = self.project(face_vector)

        if self.projections is None or len(self.labels) == 0:
            raise RuntimeError("Aucune projection d'entraînement disponible.")

        distances = np.linalg.norm(self.projections - test_projection, axis=1)
        min_index = int(np.argmin(distances))
        min_distance = float(distances[min_index])
        identity = self.labels[min_index]
        decision = "Match" if min_distance <= threshold else "No Match"

        return {
            "identity": identity,
            "distance": min_distance,
            "decision": decision,
            "bbox": bbox,
            "image": image,
            "face": face,
        }

    def annotate_result(self, image: np.ndarray, bbox, distance: float, decision: str, identity: str) -> np.ndarray:
        """
        Annote l'image test avec le rectangle de détection, la distance et la décision.
        """
        annotated = image.copy()
        if bbox is not None:
            x, y, w, h = bbox
            color = (0, 255, 0) if decision == "Match" else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        text_1 = f"Identite: {identity}"
        text_2 = f"Distance: {distance:.2f}"
        text_3 = f"Resultat: {decision}"

        cv2.putText(annotated, text_1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated, text_2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated, text_3, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if decision == "Match" else (0, 0, 255), 2)
        return annotated

    def evaluate(self, test_path: str, threshold: float) -> Dict[str, float]:
        """
        Calcule :
        - taux de reconnaissance
        - nombre de faux rejets
        - nombre de fausses acceptations

        Hypothèses :
        - le dossier test est structuré par personne comme le dossier train
        - chaque image test est une tentative authentique d'une identité connue ou inconnue
        - faux rejet : identité connue mais décision No Match
        - fausse acceptation : décision Match vers une mauvaise identité
        """
        test_dir = Path(test_path)
        if not test_dir.exists() or not test_dir.is_dir():
            raise FileNotFoundError(f"Dossier de test introuvable : {test_path}")

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        total = 0
        correct = 0
        false_rejects = 0
        false_accepts = 0
        no_face = 0

        for person_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            true_label = person_dir.name
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in valid_ext:
                    continue
                total += 1
                result = self.recognize(str(img_path), threshold=threshold)

                if result["bbox"] is None:
                    no_face += 1
                    false_rejects += 1
                    continue

                pred = result["identity"]
                decision = result["decision"]

                if decision == "Match" and pred == true_label:
                    correct += 1
                elif decision == "No Match":
                    false_rejects += 1
                else:
                    false_accepts += 1

        recognition_rate = (correct / total * 100.0) if total else 0.0
        return {
            "total_tests": total,
            "correct_matches": correct,
            "recognition_rate": recognition_rate,
            "false_rejects": false_rejects,
            "false_accepts": false_accepts,
            "no_face_detected": no_face,
        }

    def experiment_k_values(self, train_path: str, test_path: str, k_values: List[int], threshold: float) -> List[Dict[str, float]]:
        """
        Compare l'effet du nombre de composantes principales k.
        """
        results = []
        for k in k_values:
            model = FaceRecognitionPCA(n_components=k, face_size=self.face_size)
            model.train(train_path)
            metrics = model.evaluate(test_path, threshold)
            metrics["k"] = k
            results.append(metrics)
        return results

    def experiment_thresholds(self, test_path: str, thresholds: List[float]) -> List[Dict[str, float]]:
        """
        Étudie l'effet du seuil : tableau Distance vs Décision/Métriques.
        """
        results = []
        for threshold in thresholds:
            metrics = self.evaluate(test_path, threshold)
            metrics["threshold"] = threshold
            results.append(metrics)
        return results


def print_metrics_table(title: str, rows: List[Dict[str, float]], key_name: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"{key_name:>12} | {'Tx reco (%)':>12} | {'Faux rejets':>12} | {'Fausses acc.':>13} | {'Tests':>7}")
    print("-" * 70)
    for row in rows:
        print(
            f"{row[key_name]:>12} | "
            f"{row['recognition_rate']:>12.2f} | "
            f"{row['false_rejects']:>12} | "
            f"{row['false_accepts']:>13} | "
            f"{row['total_tests']:>7}"
        )


def save_annotated_image(output_path: str, image: np.ndarray) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconnaissance faciale par PCA (Eigenfaces) et Viola-Jones"
    )
    parser.add_argument("--train", required=True, help="Dossier d'apprentissage structuré par personne.")
    parser.add_argument("--test-image", required=True, help="Image test à reconnaître.")
    parser.add_argument("--test-dir", default=None, help="Dossier de test pour l'évaluation expérimentale.")
    parser.add_argument("--k", type=int, default=30, help="Nombre de composantes principales.")
    parser.add_argument("--threshold", type=float, default=2500.0, help="Seuil de décision.")
    parser.add_argument(
        "--output",
        default="resultats/resultat_test.jpg",
        help="Chemin de sauvegarde de l'image annotée.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    recognizer = FaceRecognitionPCA(n_components=args.k)
    recognizer.train(args.train)

    result = recognizer.recognize(args.test_image, threshold=args.threshold)

    print("\n--- Résultat de reconnaissance ---")
    print(f"Distance minimale : {result['distance']:.2f}")
    print(f"Identité prédite : {result['identity']}")
    print(f"Décision finale : {result['decision']}")

    annotated = recognizer.annotate_result(
        result["image"],
        result["bbox"],
        result["distance"],
        result["decision"],
        result["identity"],
    )
    save_annotated_image(args.output, annotated)
    print(f"Image annotée sauvegardée : {args.output}")

    if args.test_dir:
        metrics = recognizer.evaluate(args.test_dir, threshold=args.threshold)
        print("\n--- Évaluation globale ---")
        print(f"Nombre total de tests : {metrics['total_tests']}")
        print(f"Taux de reconnaissance : {metrics['recognition_rate']:.2f}%")
        print(f"Nombre de faux rejets : {metrics['false_rejects']}")
        print(f"Nombre de fausses acceptations : {metrics['false_accepts']}")

        k_results = recognizer.experiment_k_values(
            train_path=args.train,
            test_path=args.test_dir,
            k_values=[10, 20, 50],
            threshold=args.threshold,
        )
        print_metrics_table("Effet du nombre de composantes k", k_results, "k")

        threshold_values = [args.threshold * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]]
        threshold_results = recognizer.experiment_thresholds(args.test_dir, threshold_values)
        print_metrics_table("Effet du seuil", threshold_results, "threshold")


if __name__ == "__main__":
    main()
