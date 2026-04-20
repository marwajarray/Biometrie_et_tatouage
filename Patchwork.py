import cv2
import numpy as np
import matplotlib.pyplot as plt


KEY = 42
K = 2
N_PAIRS = 5000


def patchwork_insert(img, key, k, n_pairs):
    np.random.seed(key)
    h, w = img.shape

    watermarked = img.copy().astype(np.int32)

    coords = []
    for _ in range(n_pairs):
        x1, y1 = np.random.randint(0, h), np.random.randint(0, w)
        x2, y2 = np.random.randint(0, h), np.random.randint(0, w)

        watermarked[x1, y1] += k
        watermarked[x2, y2] -= k

        coords.append((x1, y1, x2, y2))

    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    return watermarked, coords



def patchwork_detect(img, coords):
    diffs = []
    for (x1, y1, x2, y2) in coords:
        diffs.append(int(img[x1, y1]) - int(img[x2, y2]))

    mean_diff = np.mean(diffs)
    return mean_diff



def add_noise(img):
    noise = np.random.normal(0, 5, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def jpeg_compression(img):
    cv2.imwrite("image.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)


def blur_image(img):
    return cv2.GaussianBlur(img, (5,5), 0)





img = cv2.imread("C:/Users/pc/Desktop/Patchwork/image.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Erreur: image non trouvée")
    exit()

# Insertion
watermarked, coords = patchwork_insert(img, KEY, K, N_PAIRS)

# Détection
mean_before = patchwork_detect(img, coords)
mean_after = patchwork_detect(watermarked, coords)

# Attaques
noisy = add_noise(watermarked)
compressed = jpeg_compression(watermarked)
blurred = blur_image(watermarked)

mean_noise = patchwork_detect(noisy, coords)
mean_compressed = patchwork_detect(compressed, coords)
mean_blur = patchwork_detect(blurred, coords)


print("Moyenne image originale:", mean_before)
print("Moyenne image tatouée:", mean_after)
print("Après bruit:", mean_noise)
print("Après compression:", mean_compressed)
print("Après flou:", mean_blur)

# Affichage images
plt.figure(figsize=(10,6))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2,3,2)
plt.title("Tatouée")
plt.imshow(watermarked, cmap='gray')
plt.axis('off')

plt.subplot(2,3,3)
plt.title("Bruit")
plt.imshow(noisy, cmap='gray')
plt.axis('off')

plt.subplot(2,3,4)
plt.title("Compression")
plt.imshow(compressed, cmap='gray')
plt.axis('off')

plt.subplot(2,3,5)
plt.title("Flou")
plt.imshow(blurred, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()