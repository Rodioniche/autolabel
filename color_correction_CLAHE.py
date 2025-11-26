# ================================
#  цветокоррекция CLAHE
# ================================
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

def equalize_histogram_with_mask(image, mask):
    # Преобразуем изображение в LAB цветовое пространство
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Извлекаем L компоненту
    L_channel = lab_image[:, :, 0]

    # СОХРАНЯЕМ L до обработки
    L_before = L_channel.copy()

    # эквализации CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Применяем CLAHE ко всему изображению
    L_clahe_full = clahe.apply(L_channel)

    # Вставляем только значения внутри маски
    L_channel_equalized = L_channel.copy()
    L_channel_equalized[mask] = L_clahe_full[mask]

    # СОХРАНЯЕМ L после обработки
    L_after = L_channel_equalized.copy()

    # Обновляем L канал в LAB изображении
    lab_image[:, :, 0] = L_channel_equalized

    # Преобразуем обратно в BGR
    equalized_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)


    # ОТРИСОВКА ГИСТОГРАММ L ДО И ПОСЛЕ
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("L канал до CLAHE")
    plt.hist(L_before.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.6)
    plt.xlabel("L значение")
    plt.ylabel("Количество пикселей")

    plt.subplot(1, 2, 2)
    plt.title("L канал после CLAHE")
    plt.hist(L_after.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.6)
    plt.xlabel("L значение")
    plt.ylabel("Количество пикселей")

    plt.tight_layout()
    plt.show()

    return equalized_image


def create_mask(image_path, output_path, threshold=150):
    img = Image.open(image_path).convert('RGB')
    blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
    data = np.array(blurred)

    r, g, b = data.T
    mask = (r > threshold) & (g > threshold) & (b > threshold)

    return np.logical_not(mask.T)


# Пример использования
image_path = "rectified_object.jpg"
output_path = "lab_correction_CLAHE.jpg"

mask = create_mask(image_path, output_path)

image = cv2.imread(image_path)
equalized_image = equalize_histogram_with_mask(image, mask)
cv2.imwrite(output_path, equalized_image)
