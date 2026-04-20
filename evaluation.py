
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def compute_mse(original_path: str, stego_path: str) -> float:
    """
    Calcule l'erreur quadratique moyenne (MSE) entre deux images.

    MSE = (1 / N) * Σ (original[i] - stego[i])²

    Plus le MSE est faible, plus les images sont similaires.

    Args:
        original_path: Chemin de l'image originale.
        stego_path: Chemin de l'image tatouée.

    Returns:
        La valeur du MSE.
    """
    orig = np.array(Image.open(original_path)).astype(np.float64)
    stego = np.array(Image.open(stego_path)).astype(np.float64)

    # Adapter les dimensions si nécessaire
    if orig.shape != stego.shape:
        # Convertir les deux en grayscale pour comparer
        orig = np.array(Image.open(original_path).convert('L')).astype(np.float64)
        stego = np.array(Image.open(stego_path).convert('L')).astype(np.float64)

    mse = np.mean((orig - stego) ** 2)
    return mse


def compute_psnr(original_path: str, stego_path: str) -> float:
    """
    Calcule le Peak Signal-to-Noise Ratio (PSNR) entre deux images.

    PSNR = 10 * log10(MAX² / MSE)

    où MAX = 255 pour des images 8 bits.

    Interprétation :
        - PSNR > 40 dB  → Excellente qualité (invisible)
        - PSNR 30-40 dB → Bonne qualité
        - PSNR < 30 dB  → Dégradation visible

    Args:
        original_path: Chemin de l'image originale.
        stego_path: Chemin de l'image tatouée.

    Returns:
        La valeur du PSNR en dB.
    """
    mse = compute_mse(original_path, stego_path)
    if mse == 0:
        return float('inf')  # Images identiques
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr


def compare_images(original_path: str, stego_path: str, title: str = "", save_path: str = None):
    """
    Affiche côte à côte l'image originale et l'image tatouée,
    avec l'image de différence amplifiée.

    Args:
        original_path: Chemin de l'image originale.
        stego_path: Chemin de l'image tatouée.
        title: Titre de la figure.
        save_path: Chemin pour sauvegarder la figure (optionnel).
    """
    orig = np.array(Image.open(original_path).convert('L')).astype(np.float64)
    stego = np.array(Image.open(stego_path).convert('L')).astype(np.float64)

    diff = np.abs(orig - stego)
    # Amplifier la différence pour la rendre visible
    diff_amplified = (diff * 255).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title("Image Originale")
    axes[0].axis('off')

    axes[1].imshow(stego, cmap='gray')
    axes[1].set_title("Image Tatouée")
    axes[1].axis('off')

    axes[2].imshow(diff_amplified, cmap='hot')
    axes[2].set_title("Différence (amplifiée x255)")
    axes[2].axis('off')

    mse = compute_mse(original_path, stego_path)
    psnr = compute_psnr(original_path, stego_path)

    fig.suptitle(f"{title}\nMSE = {mse:.6f} | PSNR = {psnr:.2f} dB", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EVAL] Figure sauvegardée : '{save_path}'")

    plt.show()
    plt.close()


def print_evaluation_report(original_path: str, outputs: dict):
    """
    Affiche un tableau comparatif des métriques pour toutes les méthodes.

    Args:
        original_path: Chemin de l'image originale.
        outputs: Dictionnaire {nom_méthode: chemin_image_tatouée}.
    """
    print("\n" + "=" * 65)
    print("       RAPPORT D'ÉVALUATION — Tatouage LSB")
    print("=" * 65)
    print(f"{'Méthode':<20} {'MSE':>12} {'PSNR (dB)':>12} {'Qualité':>15}")
    print("-" * 65)

    for method, path in outputs.items():
        mse = compute_mse(original_path, path)
        psnr = compute_psnr(original_path, path)

        if psnr == float('inf'):
            quality = "Identique"
        elif psnr > 40:
            quality = "Excellente"
        elif psnr > 30:
            quality = "Bonne"
        else:
            quality = "Dégradée"

        print(f"{method:<20} {mse:>12.6f} {psnr:>12.2f} {quality:>15}")

    print("=" * 65)
    print("Interprétation :")
    print("  PSNR > 40 dB  → Tatouage invisible (excellente qualité)")
    print("  PSNR 30-40 dB → Bonne qualité")
    print("  PSNR < 30 dB  → Dégradation visible")
    print()


# --------------- Test rapide ---------------
if __name__ == "__main__":
    outputs = {
        "LSB Gray": "gray_output.png",
        "LSB RGB": "rgb_output.png",
        "LSB Clé secrète": "key_output.png",
    }
    print_evaluation_report("input.png", outputs)
