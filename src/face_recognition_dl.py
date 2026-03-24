import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet


class FaceRecognitionDL:
    """
    Implémentation du TP05 :
    - Détection de visage avec MTCNN
    - Extraction d'embeddings avec FaceNet
    - Comparaison par distance euclidienne et similarité cosinus
    """

    def __init__(self, image_size: Tuple[int, int] = (160, 160), similarity_metric: str = "euclidean"):
        self.image_size = image_size
        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.database: List[Dict] = []
        self.similarity_metric = similarity_metric.lower()

    def load_image(self, image_path: str) -> np.ndarray:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

    def detect_face(self, image: np.ndarray) -> np.ndarray:
        """
        Entrée : image RGB
        Sortie : visage détecté redimensionné en 160x160
        Étapes :
        - détection MTCNN
        - extraction du visage
        - resize 160x160
        """
        detections = self.detector.detect_faces(image)
        if not detections:
            raise ValueError("Aucun visage détecté dans l'image.")

        # On choisit le visage avec la plus grande boîte englobante.
        best_detection = max(
            detections,
            key=lambda d: max(0, d["box"][2]) * max(0, d["box"][3]),
        )

        x, y, w, h = best_detection["box"]
        x, y = max(0, x), max(0, y)
        w, h = max(1, w), max(1, h)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)

        face = image[y:y2, x:x2]
        if face.size == 0:
            raise ValueError("La région du visage extraite est vide.")

        face = cv2.resize(face, self.image_size)
        return face

    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        face = face.astype("float32")
        mean, std = face.mean(), face.std()
        std = std if std > 1e-6 else 1.0
        face = (face - mean) / std
        return face

    def extract_embedding(self, face: np.ndarray) -> np.ndarray:
        """
        Entrée : image visage
        Sortie : embedding (vecteur 512 avec keras-facenet)
        Étapes :
        - normalisation
        - passage dans le réseau CNN
        """
        face = self.preprocess_face(face)
        samples = np.expand_dims(face, axis=0)
        embedding = self.embedder.embeddings(samples)[0]
        return embedding

    def build_database(self, dataset_path: str) -> None:
        """
        Parcourt un dossier :
            dataset/
                person1/
                person2/
        Pour chaque image :
        - détecter visage
        - extraire embedding
        - stocker embedding + label
        """
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dossier dataset introuvable : {dataset_path}")

        self.database = []
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        for label in sorted(os.listdir(dataset_path)):
            person_dir = os.path.join(dataset_path, label)
            if not os.path.isdir(person_dir):
                continue

            for file_name in sorted(os.listdir(person_dir)):
                if not file_name.lower().endswith(valid_extensions):
                    continue

                image_path = os.path.join(person_dir, file_name)
                try:
                    image = self.load_image(image_path)
                    face = self.detect_face(image)
                    embedding = self.extract_embedding(face)
                    self.database.append(
                        {
                            "label": label,
                            "image_path": image_path,
                            "embedding": embedding,
                        }
                    )
                except Exception as exc:
                    print(f"[WARN] Image ignorée ({image_path}) : {exc}")

        if not self.database:
            raise ValueError(
                "La base d'embeddings est vide. Vérifiez les images du dataset et la détection des visages."
            )

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcul de similarité cosinus."""
        emb1 = np.asarray(emb1)
        emb2 = np.asarray(emb2)
        denominator = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if denominator == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / denominator)

    def euclidean_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcul de distance euclidienne."""
        emb1 = np.asarray(emb1)
        emb2 = np.asarray(emb2)
        return float(np.linalg.norm(emb1 - emb2))

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray, metric: Optional[str] = None) -> float:
        metric = (metric or self.similarity_metric).lower()
        if metric == "cosine":
            return self.cosine_similarity(emb1, emb2)
        if metric == "euclidean":
            return self.euclidean_distance(emb1, emb2)
        raise ValueError("Metric non supportée. Utilisez 'euclidean' ou 'cosine'.")

    def recognize(
        self,
        image_path: str,
        threshold: float = 0.6,
        metric: Optional[str] = None,
    ) -> Tuple[str, float, str]:
        """
        Étapes :
        - charger image
        - détecter visage
        - extraire embedding
        - comparer avec base
        - trouver embedding le plus proche
        Retour : label, score, décision (Match / No Match)
        """
        if not self.database:
            raise ValueError("La base d'embeddings est vide. Exécutez build_database() d'abord.")

        metric = (metric or self.similarity_metric).lower()
        image = self.load_image(image_path)
        face = self.detect_face(image)
        query_embedding = self.extract_embedding(face)

        best_label = "Inconnu"
        best_score = None

        for sample in self.database:
            candidate_embedding = sample["embedding"]

            if metric == "euclidean":
                score = self.euclidean_distance(query_embedding, candidate_embedding)
                is_better = best_score is None or score < best_score
            elif metric == "cosine":
                score = self.cosine_similarity(query_embedding, candidate_embedding)
                is_better = best_score is None or score > best_score
            else:
                raise ValueError("Metric non supportée. Utilisez 'euclidean' ou 'cosine'.")

            if is_better:
                best_score = score
                best_label = sample["label"]

        if metric == "euclidean":
            decision = "Match" if best_score <= threshold else "No Match"
        else:
            decision = "Match" if best_score >= threshold else "No Match"

        return best_label, float(best_score), decision

    def evaluate_thresholds(self, image_pairs: List[Tuple[str, str, int]], thresholds: List[float], metric: str = "euclidean"):
        """
        image_pairs : liste de tuples (img1, img2, same_person)
        same_person = 1 si même personne, 0 sinon.
        Retourne les statistiques faux rejets / fausses acceptations par seuil.
        """
        results = []
        metric = metric.lower()

        for threshold in thresholds:
            false_rejects = 0
            false_accepts = 0
            total_same = 0
            total_diff = 0

            for img1, img2, same_person in image_pairs:
                emb1 = self.extract_embedding(self.detect_face(self.load_image(img1)))
                emb2 = self.extract_embedding(self.detect_face(self.load_image(img2)))

                if metric == "euclidean":
                    score = self.euclidean_distance(emb1, emb2)
                    predicted_match = score <= threshold
                elif metric == "cosine":
                    score = self.cosine_similarity(emb1, emb2)
                    predicted_match = score >= threshold
                else:
                    raise ValueError("Metric non supportée. Utilisez 'euclidean' ou 'cosine'.")

                if same_person == 1:
                    total_same += 1
                    if not predicted_match:
                        false_rejects += 1
                else:
                    total_diff += 1
                    if predicted_match:
                        false_accepts += 1

            results.append(
                {
                    "metric": metric,
                    "threshold": threshold,
                    "false_rejects": false_rejects,
                    "false_accepts": false_accepts,
                    "total_same": total_same,
                    "total_diff": total_diff,
                }
            )

        return results
