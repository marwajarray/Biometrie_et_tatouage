

import numpy as np
from PIL import Image
from utils import text_to_binary, binary_to_text


def embed_lsb_gray(image_path: str, message: str, output_path: str) -> None:
    """
    Insère un message dans une image en niveaux de gris via LSB.

    Étapes :
        1. Charger l'image en grayscale
        2. Convertir le message en binaire
        3. Aplatir la matrice de pixels
        4. Modifier le bit de poids faible (LSB) de chaque pixel
        5. Reconstruire et sauvegarder l'image tatouée

    Args:
        image_path: Chemin de l'image originale.
        message: Message texte à insérer.
        output_path: Chemin de sauvegarde de l'image tatouée.
    """
    # 1. Charger l'image en niveaux de gris
    img = Image.open(image_path).convert('L')
    pixels = np.array(img)
    shape = pixels.shape

    # 2. Convertir le message en binaire
    binary_msg = text_to_binary(message)
    msg_len = len(binary_msg)

    # 3. Aplatir la matrice de pixels
    flat = pixels.flatten()

    # Vérification de la capacité
    if msg_len > len(flat):
        raise ValueError(
            f"Message trop long ({msg_len} bits) pour l'image ({len(flat)} pixels)."
        )

    # 4. Modifier le LSB de chaque pixel concerné
    for i in range(msg_len):
        # Mettre le LSB à 0, puis ajouter le bit du message
        flat[i] = (flat[i] & 0xFE) | int(binary_msg[i])

    # 5. Reconstruire et sauvegarder
    stego_pixels = flat.reshape(shape)
    stego_img = Image.fromarray(stego_pixels.astype(np.uint8), mode='L')
    stego_img.save(output_path)
    print(f"[GRAY] Message inséré avec succès dans '{output_path}'")
    print(f"       Bits utilisés : {msg_len} / {len(flat)} ({msg_len/len(flat)*100:.4f}%)")


def extract_lsb_gray(image_path: str, msg_len: int) -> str:
    """
    Extrait un message caché dans une image grayscale via LSB.

    Args:
        image_path: Chemin de l'image tatouée.
        msg_len: Nombre de caractères du message à extraire.

    Returns:
        Le message extrait.
    """
    # Charger l'image tatouée
    img = Image.open(image_path).convert('L')
    pixels = np.array(img).flatten()

    # Nombre de bits à extraire
    nb_bits = msg_len * 8

    # Extraire les LSB
    binary_msg = ''
    for i in range(nb_bits):
        binary_msg += str(pixels[i] & 1)

    # Convertir en texte
    message = binary_to_text(binary_msg)
    return message


# --------------- Test rapide ---------------
if __name__ == "__main__":
    message = "bonjour"
    embed_lsb_gray("input.png", message, "gray_output.png")
    extracted = extract_lsb_gray("gray_output.png", len(message))
    print(f"[GRAY] Message extrait : '{extracted}'")
    print(f"[GRAY] Correspondance  : {'✓ OK' if extracted == message else '✗ ERREUR'}")
