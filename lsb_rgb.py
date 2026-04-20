
import numpy as np
from PIL import Image
from utils import text_to_binary, binary_to_text


def embed_lsb_rgb(image_path: str, message: str, output_path: str) -> None:
    """
    Insère un message dans une image RGB via LSB sur les 3 canaux.

    Avantage par rapport au grayscale :
        - Capacité x3 (on utilise R, G et B de chaque pixel)

    Étapes :
        1. Charger l'image en RGB
        2. Convertir le message en binaire
        3. Parcourir les pixels et insérer un bit par canal (R, G, B)
        4. Sauvegarder l'image tatouée

    Args:
        image_path: Chemin de l'image originale.
        message: Message texte à insérer.
        output_path: Chemin de sauvegarde de l'image tatouée.
    """
    # 1. Charger l'image en RGB
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img)
    shape = pixels.shape  # (H, W, 3)

    # 2. Convertir le message en binaire
    binary_msg = text_to_binary(message)
    msg_len = len(binary_msg)

    # 3. Aplatir les canaux
    flat = pixels.flatten()  # Chaque élément = une composante (R, G ou B)

    # Vérification de la capacité
    if msg_len > len(flat):
        raise ValueError(
            f"Message trop long ({msg_len} bits) pour l'image ({len(flat)} composantes)."
        )

    # 4. Modifier le LSB de chaque composante concernée
    for i in range(msg_len):
        flat[i] = (flat[i] & 0xFE) | int(binary_msg[i])

    # 5. Reconstruire et sauvegarder
    stego_pixels = flat.reshape(shape)
    stego_img = Image.fromarray(stego_pixels.astype(np.uint8), mode='RGB')
    stego_img.save(output_path)

    total_capacity = shape[0] * shape[1] * 3
    print(f"[RGB]  Message inséré avec succès dans '{output_path}'")
    print(f"       Bits utilisés : {msg_len} / {total_capacity} ({msg_len/total_capacity*100:.4f}%)")
    print(f"       Capacité RGB vs Gray : x3 (3 canaux utilisés)")


def extract_lsb_rgb(image_path: str, msg_len: int) -> str:
    """
    Extrait un message caché dans une image RGB via LSB.

    Args:
        image_path: Chemin de l'image tatouée.
        msg_len: Nombre de caractères du message à extraire.

    Returns:
        Le message extrait.
    """
    # Charger l'image tatouée
    img = Image.open(image_path).convert('RGB')
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
    embed_lsb_rgb("input.png", message, "rgb_output.png")
    extracted = extract_lsb_rgb("rgb_output.png", len(message))
    print(f"[RGB]  Message extrait : '{extracted}'")
    print(f"[RGB]  Correspondance  : {'✓ OK' if extracted == message else '✗ ERREUR'}")
