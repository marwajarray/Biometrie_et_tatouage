"""
TP01 : Traitement des images avec PIL (Pillow)
Auteur  : Hamdi Chebbi
Cours   : Biométrie & Tatouage  —  ING-4-SSIRF
"""

import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

# ─────────────────────────────────────────────
#  CONFIGURATION GLOBALE
# ─────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_PATH = "image.jpg"   # <-- remplacer par le chemin de votre image


def load_image(path: str) -> Image.Image:
    """Charge et retourne une image PIL depuis le disque."""
    img = Image.open(path)
    return img


# ════════════════════════════════════════════════════════════
# PARTIE 1 — Lecture et affichage de l'image originale
# Objectif : Charger une image couleur et l'afficher telle quelle.
# Effet observé : On visualise l'image brute sans aucune modification.
# ════════════════════════════════════════════════════════════
def partie1_lecture(img: Image.Image):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Image originale")
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "image_originale.png")
    img.save(save_path)
    plt.savefig(os.path.join(RESULTS_DIR, "partie1_affichage.png"))
    plt.close()
    print(f"[Partie 1] Sauvegardée → {save_path}")


# ════════════════════════════════════════════════════════════
# PARTIE 2 — Redimensionnement
# Objectif : Réduire ou modifier les dimensions de l'image.
# Effet observé : L'image est redimensionnée à 200×200 pixels ;
#                 les proportions peuvent être altérées selon la taille choisie.
# ════════════════════════════════════════════════════════════
def partie2_redimensionnement(img: Image.Image):
    taille = (200, 200)          # essayez aussi (300, 150)
    img_resized = img.resize(taille)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(f"Originale  {img.size[0]}×{img.size[1]}")
    axes[1].imshow(img_resized)
    axes[1].axis("off")
    axes[1].set_title(f"Redimensionnée  {taille[0]}×{taille[1]}")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "image_redimensionnee.png")
    img_resized.save(save_path)
    plt.savefig(os.path.join(RESULTS_DIR, "partie2_comparaison.png"))
    plt.close()
    print(f"[Partie 2] Sauvegardée → {save_path}")


# ════════════════════════════════════════════════════════════
# PARTIE 3 — Ajustement de la luminosité
# Objectif : Augmenter la luminosité globale de l'image.
# Effet observé : L'image apparaît plus claire ; un facteur > 1 éclaircit,
#                 un facteur < 1 assombrit.
# ════════════════════════════════════════════════════════════
def partie3_luminosite(img: Image.Image):
    facteur = 1.5
    enhancer = ImageEnhance.Brightness(img)
    img_bright = enhancer.enhance(facteur)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Originale")
    axes[1].imshow(img_bright)
    axes[1].axis("off")
    axes[1].set_title(f"Luminosité × {facteur}")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "image_luminosite_augmente.png")
    img_bright.save(save_path)
    plt.savefig(os.path.join(RESULTS_DIR, "partie3_comparaison.png"))
    plt.close()
    print(f"[Partie 3] Sauvegardée → {save_path}")


# ════════════════════════════════════════════════════════════
# PARTIE 4 — Conversion en niveaux de gris
# Objectif : Supprimer l'information couleur pour simplifier l'image.
# Effet observé : L'image passe de 3 canaux RGB à 1 canal (L) ;
#                 chaque pixel représente une intensité lumineuse.
# ════════════════════════════════════════════════════════════
def partie4_niveaux_gris(img: Image.Image) -> Image.Image:
    img_gray = img.convert("L")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Originale (RGB)")
    axes[1].imshow(img_gray, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Niveaux de gris (L)")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "image_gris.png")
    img_gray.save(save_path)
    plt.savefig(os.path.join(RESULTS_DIR, "partie4_comparaison.png"))
    plt.close()
    print(f"[Partie 4] Sauvegardée → {save_path}")
    return img_gray


# ════════════════════════════════════════════════════════════
# PARTIE 5 — Binarisation
# Objectif : Convertir l'image en deux niveaux (noir/blanc) via un seuil.
# Effet observé : Les pixels > seuil deviennent blancs (255), les autres
#                 deviennent noirs (0) — utile pour la segmentation.
# ════════════════════════════════════════════════════════════
def partie5_binarisation(img_gray: Image.Image):
    seuil = 128                  # modifiez entre 0 et 255
    img_bin = img_gray.point(lambda p: 255 if p > seuil else 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Niveaux de gris")
    axes[1].imshow(img_bin, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title(f"Binarisée (seuil = {seuil})")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "image_binarisee.png")
    img_bin.save(save_path)
    plt.savefig(os.path.join(RESULTS_DIR, "partie5_comparaison.png"))
    plt.close()
    print(f"[Partie 5] Sauvegardée → {save_path}")


# ════════════════════════════════════════════════════════════
# PARTIE 6 — Détection des contours
# Objectif : Mettre en évidence les bords et transitions dans l'image.
# Effet observé : Les contours des objets ressortent en blanc sur fond
#                 sombre grâce au filtre FIND_EDGES.
# ════════════════════════════════════════════════════════════
def partie6_contours(img_gray: Image.Image):
    img_edges = img_gray.filter(ImageFilter.FIND_EDGES)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Niveaux de gris")
    axes[1].imshow(img_edges, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Détection des contours")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "image_contours.png")
    img_edges.save(save_path)
    plt.savefig(os.path.join(RESULTS_DIR, "partie6_comparaison.png"))
    plt.close()
    print(f"[Partie 6] Sauvegardée → {save_path}")


# ════════════════════════════════════════════════════════════
# PARTIE 7 — Filtrage et débruitage (Flou Gaussien)
# Objectif : Atténuer le bruit en appliquant un lissage gaussien.
# Effet observé : Plus le rayon est grand, plus l'image est floue ;
#                 on compare radius 1, 2 et 3 côte à côte.
# ════════════════════════════════════════════════════════════
def partie7_flou_gaussien(img: Image.Image):
    rayons = [1, 2, 3]
    images_floues = [img.filter(ImageFilter.GaussianBlur(radius=r)) for r in rayons]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Originale")
    for i, (r, im) in enumerate(zip(rayons, images_floues)):
        axes[i + 1].imshow(im)
        axes[i + 1].axis("off")
        axes[i + 1].set_title(f"Flou Gaussien r={r}")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "image_flou_gaussien.png")
    images_floues[-1].save(save_path)          # sauvegarde du rayon maximal
    plt.savefig(os.path.join(RESULTS_DIR, "partie7_comparaison.png"))
    plt.close()
    print(f"[Partie 7] Sauvegardée → {save_path}")


# ════════════════════════════════════════════════════════════
# PARTIE 8 — Histogramme de l'image
# Objectif : Visualiser la distribution des niveaux de gris.
# Effet observé : La courbe montre la fréquence de chaque intensité
#                 (0 = noir, 255 = blanc) ; utile pour analyser le contraste.
# ════════════════════════════════════════════════════════════
def partie8_histogramme(img_gray: Image.Image):
    histo = img_gray.histogram()   # liste de 256 valeurs
    niveaux = list(range(256))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Image en niveaux de gris")

    axes[1].plot(niveaux, histo, color="dimgray")
    axes[1].set_title("Histogramme des niveaux de gris")
    axes[1].set_xlabel("Niveau d'intensité (0–255)")
    axes[1].set_ylabel("Nombre de pixels")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(RESULTS_DIR, "partie8_histogramme.png"))
    plt.close()
    print("[Partie 8] Histogramme sauvegardé → results/partie8_histogramme.png")


# ════════════════════════════════════════════════════════════
# PARTIE 9 — Égalisation de l'histogramme
# Objectif : Améliorer le contraste en redistribuant les intensités.
# Effet observé : L'histogramme s'étale uniformément sur 0–255 ;
#                 les détails dans les zones sombres ou claires deviennent visibles.
# ════════════════════════════════════════════════════════════
def partie9_egalisation(img_gray: Image.Image):
    img_eq = ImageOps.equalize(img_gray)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Avant égalisation")
    axes[1].imshow(img_eq, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Après égalisation")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "image_egalisee.png")
    img_eq.save(save_path)
    plt.savefig(os.path.join(RESULTS_DIR, "partie9_comparaison.png"))
    plt.close()
    print(f"[Partie 9] Sauvegardée → {save_path}")


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TP01 : Traitement des images avec PIL ===\n")

    # Chargement de l'image
    img = load_image(IMAGE_PATH)
    print(f"Image chargée : {IMAGE_PATH}  ({img.size[0]}×{img.size[1]} px, mode={img.mode})\n")

    # Exécution de toutes les parties
    partie1_lecture(img)
    partie2_redimensionnement(img)
    partie3_luminosite(img)
    img_gray = partie4_niveaux_gris(img)
    partie5_binarisation(img_gray)
    partie6_contours(img_gray)
    partie7_flou_gaussien(img)
    partie8_histogramme(img_gray)
    partie9_egalisation(img_gray)

    print("\n✅ Toutes les parties ont été exécutées.")
    print(f"   Résultats disponibles dans le dossier : {RESULTS_DIR}/")
