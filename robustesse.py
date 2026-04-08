

import numpy as np
from PIL import Image, ImageFilter
from lsb_gray import extract_lsb_gray
from lsb_rgb import extract_lsb_rgb
from lsb_key import extract_lsb_key


def attack_compression_jpeg(image_path: str, output_path: str, quality: int = 75) -> str:
    """
    Attaque par compression JPEG.
    La compression JPEG est avec perte et modifie les LSB.

    Args:
        image_path: Image tatouée d'entrée.
        output_path: Image après compression.
        quality: Qualité JPEG (1-100).

    Returns:
        Chemin de l'image compressée.
    """
    img = Image.open(image_path)
    # Sauvegarder en JPEG puis recharger
    jpeg_path = output_path.replace('.png', '.jpg')
    img.save(jpeg_path, 'JPEG', quality=quality)
    # Reconvertir en PNG pour l'extraction
    img_jpeg = Image.open(jpeg_path)
    img_jpeg.save(output_path)
    print(f"[ATK]  Compression JPEG (qualité={quality}) → '{output_path}'")
    return output_path


def attack_gaussian_noise(image_path: str, output_path: str, sigma: float = 5.0) -> str:
    """
    Attaque par ajout de bruit gaussien.
    Le bruit modifie aléatoirement les valeurs des pixels.

    Args:
        image_path: Image tatouée d'entrée.
        output_path: Image bruitée.
        sigma: Écart-type du bruit gaussien.

    Returns:
        Chemin de l'image bruitée.
    """
    img = np.array(Image.open(image_path)).astype(np.float64)
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    Image.fromarray(noisy).save(output_path)
    print(f"[ATK]  Bruit gaussien (σ={sigma}) → '{output_path}'")
    return output_path


def attack_median_filter(image_path: str, output_path: str, size: int = 3) -> str:
    """
    Attaque par filtre médian.
    Le filtrage médian remplace chaque pixel par la médiane de ses voisins.

    Args:
        image_path: Image tatouée d'entrée.
        output_path: Image filtrée.
        size: Taille du noyau du filtre médian.

    Returns:
        Chemin de l'image filtrée.
    """
    img = Image.open(image_path)
    filtered = img.filter(ImageFilter.MedianFilter(size=size))
    filtered.save(output_path)
    print(f"[ATK]  Filtre médian (taille={size}) → '{output_path}'")
    return output_path


def attack_resize(image_path: str, output_path: str, scale: float = 0.5) -> str:
    """
    Attaque par redimensionnement (réduction puis agrandissement).
    La perte d'information lors du redimensionnement détruit les LSB.

    Args:
        image_path: Image tatouée d'entrée.
        output_path: Image après redimensionnement.
        scale: Facteur de réduction.

    Returns:
        Chemin de l'image redimensionnée.
    """
    img = Image.open(image_path)
    w, h = img.size
    # Réduire
    small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    # Ré-agrandir à la taille originale
    resized = small.resize((w, h), Image.BILINEAR)
    resized.save(output_path)
    print(f"[ATK]  Redimensionnement (×{scale} puis retour) → '{output_path}'")
    return output_path


def attack_crop_paste(image_path: str, output_path: str, crop_ratio: float = 0.1) -> str:
    """
    Attaque par recadrage (crop) d'une partie de l'image.
    Remplace une portion de l'image par du blanc.

    Args:
        image_path: Image tatouée d'entrée.
        output_path: Image après recadrage.
        crop_ratio: Proportion de l'image modifiée.

    Returns:
        Chemin de l'image modifiée.
    """
    img = Image.open(image_path)
    pixels = np.array(img)
    h, w = pixels.shape[:2]
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    # Remplacer un coin par du blanc
    pixels[:crop_h, :crop_w] = 255
    Image.fromarray(pixels).save(output_path)
    print(f"[ATK]  Recadrage ({crop_ratio*100:.0f}% coin supérieur gauche) → '{output_path}'")
    return output_path


def test_robustesse(message: str, msg_len: int):
    """
    Teste la robustesse du tatouage LSB face à différentes attaques.
    
    Args:
        message: Message original inséré.
        msg_len: Longueur du message.
    """
    print("\n" + "=" * 70)
    print("       TEST DE ROBUSTESSE — Tatouage LSB Gray")
    print("=" * 70)

    attacks = {
        "JPEG Q=75": lambda: attack_compression_jpeg("gray_output.png", "atk_jpeg.png", quality=75),
        "JPEG Q=50": lambda: attack_compression_jpeg("gray_output.png", "atk_jpeg50.png", quality=50),
        "Bruit σ=5": lambda: attack_gaussian_noise("gray_output.png", "atk_noise5.png", sigma=5),
        "Bruit σ=15": lambda: attack_gaussian_noise("gray_output.png", "atk_noise15.png", sigma=15),
        "Filtre médian 3×3": lambda: attack_median_filter("gray_output.png", "atk_median.png", size=3),
        "Redim. ×0.5": lambda: attack_resize("gray_output.png", "atk_resize.png", scale=0.5),
        "Recadrage 10%": lambda: attack_crop_paste("gray_output.png", "atk_crop.png", crop_ratio=0.1),
    }

    print(f"\n{'Attaque':<25} {'Message extrait':<20} {'Correct ?':>10}")
    print("-" * 60)

    for attack_name, attack_func in attacks.items():
        try:
            atk_path = attack_func()
            extracted = extract_lsb_gray(atk_path, msg_len)
            ok = "✓ OUI" if extracted == message else "✗ NON"
            display_msg = extracted if len(extracted) <= 15 else extracted[:15] + "..."
            print(f"{attack_name:<25} '{display_msg}'{'':>5} {ok:>10}")
        except Exception as e:
            print(f"{attack_name:<25} ERREUR: {str(e)[:30]}")

    print("=" * 70)
    print("Conclusion : Le tatouage LSB est FRAGILE face à toutes ces attaques.")
    print("Le LSB n'est pas conçu pour la robustesse mais pour la capacité et")
    print("l'invisibilité. Pour la robustesse, on utilise des méthodes dans")
    print("le domaine fréquentiel (DCT, DWT).")
    print()


# --------------- Test rapide ---------------
if __name__ == "__main__":
    test_robustesse("bonjour", 7)
