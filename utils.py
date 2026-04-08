


def text_to_binary(text: str) -> str:
    """
    Convertit une chaîne de caractères en une séquence binaire.
    Chaque caractère est encodé sur 8 bits (ASCII).

    Args:
        text: Le message texte à convertir.

    Returns:
        Une chaîne de '0' et '1' représentant le message en binaire.

    Exemple:
        >>> text_to_binary("A")
        '01000001'
    """
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary


def binary_to_text(binary: str) -> str:
    """
    Convertit une séquence binaire en texte ASCII.
    Découpe la chaîne binaire en blocs de 8 bits.

    Args:
        binary: Chaîne binaire à convertir.

    Returns:
        Le message texte reconstruit.

    Exemple:
        >>> binary_to_text('01000001')
        'A'
    """
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    return text


# --------------- Test rapide ---------------
if __name__ == "__main__":
    msg = "bonjour"
    b = text_to_binary(msg)
    print(f"Texte original  : {msg}")
    print(f"Binaire         : {b}")
    print(f"Longueur binaire: {len(b)} bits")
    print(f"Texte reconverti: {binary_to_text(b)}")
