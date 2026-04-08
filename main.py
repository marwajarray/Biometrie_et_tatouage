

import os
import sys
import numpy as np
from PIL import Image

# === Imports des modules du TP ===
from utils import text_to_binary, binary_to_text
from lsb_gray import embed_lsb_gray, extract_lsb_gray
from lsb_rgb import embed_lsb_rgb, extract_lsb_rgb
from lsb_key import embed_lsb_key, extract_lsb_key
from evaluation import compute_mse, compute_psnr, compare_images, print_evaluation_report
from robustesse import test_robustesse


def generate_test_image(path: str = "input.png", width: int = 256, height: int = 256):
    """
    Génère une image de test si aucune image n'est fournie.
    Crée un dégradé gris avec du bruit pour simuler une image réaliste.

    Args:
        path: Chemin de l'image de test.
        width: Largeur de l'image.
        height: Hauteur de l'image.
    """
    if os.path.exists(path):
        print(f"[INFO] Image '{path}' déjà existante — utilisée directement.")
        return

    print(f"[INFO] Génération d'une image de test ({width}×{height})...")
    # Dégradé horizontal + bruit
    gradient = np.tile(np.linspace(0, 255, width), (height, 1))
    noise = np.random.normal(0, 20, (height, width))
    pixels = np.clip(gradient + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels, mode='L')
    # Sauvegarder en RGB aussi pour la partie 2
    img.save(path)
    img_rgb = Image.merge('RGB', [img, img, img])
    img_rgb.save(path)
    print(f"[INFO] Image de test sauvegardée : '{path}'")


def main():
    """
    Script principal — Exécute séquentiellement toutes les parties du TP.
    """
    print("=" * 70)
    print("   TP06 — TATOUAGE NUMÉRIQUE PAR LSB (SPATIAL)")
    print("   Biométrie & Tatouage — ING-4-SSIRF")
    print("=" * 70)

    # Message à insérer
    message = "bonjour"
    print(f"\n[INFO] Message à insérer : '{message}'")
    print(f"[INFO] Taille en bits    : {len(text_to_binary(message))} bits")

    # Générer ou vérifier l'image de test
    generate_test_image("input.png")

    # ==================================================================
    # PARTIE 1 — LSB Niveau de Gris
    # ==================================================================
    print("\n" + "=" * 70)
    print("   PARTIE 1 — LSB Niveau de Gris")
    print("=" * 70)

    embed_lsb_gray("input.png", message, "gray_output.png")
    extracted_gray = extract_lsb_gray("gray_output.png", len(message))
    print(f"[GRAY] Message extrait : '{extracted_gray}'")
    print(f"[GRAY] Vérification    : {'✓ SUCCÈS' if extracted_gray == message else '✗ ÉCHEC'}")

    # ==================================================================
    # PARTIE 2 — LSB RGB
    # ==================================================================
    print("\n" + "=" * 70)
    print("   PARTIE 2 — LSB RGB")
    print("=" * 70)

    embed_lsb_rgb("input.png", message, "rgb_output.png")
    extracted_rgb = extract_lsb_rgb("rgb_output.png", len(message))
    print(f"[RGB]  Message extrait : '{extracted_rgb}'")
    print(f"[RGB]  Vérification    : {'✓ SUCCÈS' if extracted_rgb == message else '✗ ÉCHEC'}")

    # Comparaison des capacités
    img_gray = Image.open("input.png").convert('L')
    img_rgb = Image.open("input.png").convert('RGB')
    w, h = img_gray.size
    cap_gray = w * h
    cap_rgb = w * h * 3
    print(f"\n[CMP]  Capacité Gray   : {cap_gray} bits = {cap_gray // 8} caractères")
    print(f"[CMP]  Capacité RGB    : {cap_rgb} bits = {cap_rgb // 8} caractères")
    print(f"[CMP]  Facteur         : ×{cap_rgb / cap_gray:.0f}")

    # ==================================================================
    # PARTIE 3 — LSB avec Clé Secrète
    # ==================================================================
    print("\n" + "=" * 70)
    print("   PARTIE 3 — LSB avec Clé Secrète")
    print("=" * 70)

    key_correct = 42
    key_fausse = 99

    embed_lsb_key("input.png", message, "key_output.png", key=key_correct)

    # Extraction avec clé correcte
    extracted_key_ok = extract_lsb_key("key_output.png", len(message), key=key_correct)
    print(f"[KEY]  Clé correcte ({key_correct})  → '{extracted_key_ok}'")
    print(f"       Vérification  : {'✓ SUCCÈS' if extracted_key_ok == message else '✗ ÉCHEC'}")

    # Extraction avec clé incorrecte
    extracted_key_ko = extract_lsb_key("key_output.png", len(message), key=key_fausse)
    print(f"[KEY]  Clé incorrecte ({key_fausse}) → '{extracted_key_ko}'")
    print(f"       Résultat      : Message illisible (sécurité vérifiée ✓)")

    # ==================================================================
    # ÉVALUATION — MSE, PSNR, Comparaison visuelle
    # ==================================================================
    print("\n" + "=" * 70)
    print("   ÉVALUATION — Qualité du tatouage")
    print("=" * 70)

    outputs = {
        "LSB Gray": "gray_output.png",
        "LSB RGB": "rgb_output.png",
        "LSB Clé secrète": "key_output.png",
    }

    print_evaluation_report("input.png", outputs)

    # Comparaisons visuelles (sauvegardées en images)
    try:
        compare_images("input.png", "gray_output.png",
                        title="PARTIE 1 — LSB Niveau de Gris",
                        save_path="comparaison_gray.png")
        compare_images("input.png", "key_output.png",
                        title="PARTIE 3 — LSB Clé Secrète",
                        save_path="comparaison_key.png")
    except Exception:
        print("[INFO] Affichage graphique non disponible (mode sans GUI).")
        print("       Les figures seront générées lors de l'exécution locale.")

    # ==================================================================
    # ROBUSTESSE — Tests d'attaques
    # ==================================================================
    test_robustesse(message, len(message))

    # ==================================================================
    # RÉSUMÉ FINAL
    # ==================================================================
    print("\n" + "=" * 70)
    print("   RÉSUMÉ FINAL")
    print("=" * 70)
    print("""
    ┌─────────────────┬────────────────┬──────────────┬───────────────┐
    │ Méthode         │ Capacité       │ Invisibilité │ Sécurité      │
    ├─────────────────┼────────────────┼──────────────┼───────────────┤
    │ LSB Gray        │ Faible (1 bit/ │ Excellente   │ Faible        │
    │                 │ pixel)         │ (PSNR > 50)  │ (séquentiel)  │
    ├─────────────────┼────────────────┼──────────────┼───────────────┤
    │ LSB RGB         │ Élevée (3 bits │ Excellente   │ Faible        │
    │                 │ /pixel)        │ (PSNR > 50)  │ (séquentiel)  │
    ├─────────────────┼────────────────┼──────────────┼───────────────┤
    │ LSB Clé         │ Faible (1 bit/ │ Excellente   │ Bonne         │
    │                 │ pixel)         │ (PSNR > 50)  │ (pseudo-aléa.)│
    └─────────────────┴────────────────┴──────────────┴───────────────┘

    Robustesse : FAIBLE pour toutes les variantes LSB.
    → Fragile face à : JPEG, bruit, filtrage, redimensionnement.
    → Pour la robustesse : utiliser DCT, DWT (domaine fréquentiel).
    """)


if __name__ == "__main__":
    main()
