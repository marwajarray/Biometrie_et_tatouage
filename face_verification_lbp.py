#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import euclidean


class FaceVerificationSystem:
    """
    Système de vérification faciale par Viola-Jones + LBP.
    """

    def __init__(self, face_size: Tuple[int, int] = (128, 128)):
        """
        Initialisation du détecteur de visage et des variables internes.
        """
        self.face_size = face_size
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.detector.empty():
            raise RuntimeError("Impossible de charger haarcascade_frontalface_default.xml")

        self.reference_face: Optional[np.ndarray] = None
        self.reference_features: Optional[np.ndarray] = None
        self.reference_image_path: Optional[str] = None

    def detect_face(self, image: np.ndarray):
        """
        Détection et retour des coordonnées du visage.
        Si plusieurs visages sont détectés, seul le plus grand est conservé.

        Retourne:
            face_resized (grayscale 128x128), bbox(x, y, w, h), gray_image
        """
        if image is None:
            return None, None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return None, None, gray

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, self.face_size, interpolation=cv2.INTER_AREA)
        return face_resized, (int(x), int(y), int(w), int(h)), gray

    def extract_lbp_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extraction de l'histogramme LBP normalisé (256 bins).
        Principe:
            - comparaison du pixel central avec ses 8 voisins immédiats
            - voisin >= centre => bit = 1, sinon 0
        """
        if face_image is None:
            raise ValueError("L'image de visage fournie est vide.")

        if len(face_image.shape) != 2:
            raise ValueError("L'image du visage doit être en niveaux de gris.")

        h, w = face_image.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        center = face_image[1:-1, 1:-1]

        neighbors = [
            face_image[0:-2, 0:-2],  # haut-gauche
            face_image[0:-2, 1:-1],  # haut
            face_image[0:-2, 2:  ],  # haut-droite
            face_image[1:-1, 2:  ],  # droite
            face_image[2:  , 2:  ],  # bas-droite
            face_image[2:  , 1:-1],  # bas
            face_image[2:  , 0:-2],  # bas-gauche
            face_image[1:-1, 0:-2],  # gauche
        ]

        for idx, neighbor in enumerate(neighbors):
            lbp |= ((neighbor >= center).astype(np.uint8) << (7 - idx))

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-12)
        return hist

    def setup_reference(self, image_path: str) -> Dict[str, object]:
        """
        Charge l'image de référence, détecte le visage et stocke les features LBP.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Impossible de lire l'image de référence : {image_path}")

        face, bbox, _ = self.detect_face(image)
        if face is None:
            raise ValueError("Aucun visage détecté dans l'image de référence.")

        self.reference_face = face
        self.reference_features = self.extract_lbp_features(face)
        self.reference_image_path = image_path

        return {
            "image": image,
            "bbox": bbox,
            "face": face,
            "features": self.reference_features,
        }

    def verify_face(self, image_path: str, threshold: float = 0.75) -> Dict[str, object]:
        """
        Vérifie l'image test par rapport à l'image de référence.

        Retourne:
            - distance
            - similarité = 1 - distance
            - décision (Match / No Match)
        """
        if self.reference_features is None:
            raise RuntimeError("Aucune référence configurée. Appelez d'abord setup_reference().")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Impossible de lire l'image test : {image_path}")

        face, bbox, _ = self.detect_face(image)
        if face is None:
            return {
                "image": image,
                "bbox": None,
                "distance": float("inf"),
                "similarity": float("-inf"),
                "decision": "No Match",
                "face": None,
            }

        test_features = self.extract_lbp_features(face)
        distance = float(euclidean(self.reference_features, test_features))
        similarity = float(1.0 - distance)
        decision = "Match" if similarity >= threshold else "No Match"

        return {
            "image": image,
            "bbox": bbox,
            "distance": distance,
            "similarity": similarity,
            "decision": decision,
            "face": face,
            "features": test_features,
        }

    def annotate_image(
        self,
        image: np.ndarray,
        bbox,
        title: str,
        decision: Optional[str] = None,
        similarity: Optional[float] = None,
    ) -> np.ndarray:
        """
        Dessine le rectangle de détection et le texte de résultat sur l'image.
        """
        annotated = image.copy()
        color = (255, 200, 0)

        if decision is not None:
            color = (0, 255, 0) if decision == "Match" else (0, 0, 255)

        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        cv2.putText(annotated, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if similarity is not None:
            cv2.putText(
                annotated,
                f"Similarite: {similarity:.4f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        if decision is not None:
            cv2.putText(
                annotated,
                f"Decision: {decision}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return annotated

    def save_side_by_side(self, ref_img: np.ndarray, test_img: np.ndarray, output_path: str) -> None:
        """
        Sauvegarde les images référence/test côte à côte.
        """
        h = max(ref_img.shape[0], test_img.shape[0])

        def pad_to_height(img, target_h):
            if img.shape[0] == target_h:
                return img
            pad = target_h - img.shape[0]
            return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        ref_img = pad_to_height(ref_img, h)
        test_img = pad_to_height(test_img, h)
        combined = np.hstack((ref_img, test_img))

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), combined)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TP03 - Vérification faciale par LBP et Viola-Jones"
    )
    parser.add_argument("--reference", required=True, help="Chemin vers l'image de référence")
    parser.add_argument("--test", required=True, help="Chemin vers l'image de test")
    parser.add_argument("--threshold", type=float, default=0.75, help="Seuil de décision")
    parser.add_argument(
        "--output",
        default="resultats/verification_lbp.jpg",
        help="Chemin de sauvegarde de l'image annotée",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Afficher les images avec OpenCV (si l'environnement le permet)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    system = FaceVerificationSystem()

    ref_result = system.setup_reference(args.reference)
    test_result = system.verify_face(args.test, threshold=args.threshold)

    print("\n--- Résultat de vérification faciale ---")
    print(f"Distance euclidienne : {test_result['distance']:.6f}")
    print(f"Similarité : {test_result['similarity']:.6f}")
    print(f"Décision : {test_result['decision']}")

    ref_annotated = system.annotate_image(
        ref_result["image"],
        ref_result["bbox"],
        title="Image de reference",
    )
    test_annotated = system.annotate_image(
        test_result["image"],
        test_result["bbox"],
        title="Image de test",
        decision=test_result["decision"],
        similarity=test_result["similarity"],
    )

    system.save_side_by_side(ref_annotated, test_annotated, args.output)
    print(f"Image résultat sauvegardée : {args.output}")

    if args.display:
        cv2.imshow("Reference", ref_annotated)
        cv2.imshow("Test", test_annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
