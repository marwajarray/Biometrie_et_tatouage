

import numpy as np
from PIL import Image
from utils import text_to_binary, binary_to_text


def embed_lsb_key(image_path: str, message: str, output_path: str, key: int = 42) -> None:
    """
    Insère un message dans une image en utilisant des positions pseudo-aléatoires
    générées par une clé secrète (seed).

    Principe de sécurité :
        - Au lieu d'insérer les bits séquentiellement (pixels 0, 1, 2, ...),
          on utilise un générateur pseudo-aléatoire initialisé avec la clé
          pour choisir les positions de manière imprévisible.
        - Sans la clé correcte, l'extraction est impossible.

    Args:
        image_path: Chemin de l'image originale.
        message: Message texte à insérer.
        output_path: Chemin de sauvegarde de l'image tatouée.
        key: Clé secrète (seed du générateur aléatoire).
    """
    # 1. Charger l'image en niveaux de gris
    img = Image.open(image_path).convert('L')
    pixels = np.array(img)
    shape = pixels.shape
    flat = pixels.flatten()

    # 2. Convertir le message en binaire
    binary_msg = text_to_binary(message)
    msg_len = len(binary_msg)

    # Vérification de la capacité
    if msg_len > len(flat):
        raise ValueError(
            f"Message trop long ({msg_len} bits) pour l'image ({len(flat)} pixels)."
        )

    # 3. Générer les positions pseudo-aléatoires avec la clé
    rng = np.random.RandomState(key)
    positions = rng.permutation(len(flat))[:msg_len]

    # 4. Insérer les bits aux positions choisies
    for i, pos in enumerate(positions):
        flat[pos] = (flat[pos] & 0xFE) | int(binary_msg[i])

    # 5. Reconstruire et sauvegarder
    stego_pixels = flat.reshape(shape)
    stego_img = Image.fromarray(stego_pixels.astype(np.uint8), mode='L')
    stego_img.save(output_path)
    print(f"[KEY]  Message inséré avec clé={key} dans '{output_path}'")
    print(f"       Positions pseudo-aléatoires : {positions[:5]}... (affichage partiel)")


def extract_lsb_key(image_path: str, msg_len: int, key: int = 42) -> str:
    """
    Extrait un message caché avec LSB sécurisé par clé.

    La même clé doit être utilisée pour retrouver les positions exactes
    des bits insérés.

    Args:
        image_path: Chemin de l'image tatouée.
        msg_len: Nombre de caractères du message à extraire.
        key: Clé secrète (doit être identique à celle de l'insertion).

    Returns:
        Le message extrait (correct si la clé est bonne, charabia sinon).
    """
    # Charger l'image tatouée
    img = Image.open(image_path).convert('L')
    pixels = np.array(img).flatten()

    # Nombre de bits à extraire
    nb_bits = msg_len * 8

    # Régénérer les mêmes positions avec la même clé
    rng = np.random.RandomState(key)
    positions = rng.permutation(len(pixels))[:nb_bits]

    # Extraire les LSB aux positions
    binary_msg = ''
    for pos in positions:
        binary_msg += str(pixels[pos] & 1)

    # Convertir en texte
    message = binary_to_text(binary_msg)
    return message


# --------------- Test rapide ---------------
if __name__ == "__main__":
    message = "bonjour"
    key_correct = 42
    key_fausse = 99

    # Insertion avec clé correcte
    embed_lsb_key("input.png", message, "key_output.png", key=key_correct)

    # Extraction avec clé correcte
    extracted_ok = extract_lsb_key("key_output.png", len(message), key=key_correct)
    print(f"[KEY]  Clé correcte ({key_correct}) → Message : '{extracted_ok}'")
    print(f"       Correspondance : {'✓ OK' if extracted_ok == message else '✗ ERREUR'}")

    # Extraction avec clé incorrecte
    extracted_ko = extract_lsb_key("key_output.png", len(message), key=key_fausse)
    print(f"[KEY]  Clé incorrecte ({key_fausse}) → Message : '{extracted_ko}'")
    print(f"       Résultat attendu : charabia (extraction impossible sans la bonne clé)")
