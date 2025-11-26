# ================================
#  цветокоррекция обычная
# ================================

# ================================
#  Установка зависимостей
# ================================
#!pip install cv2
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

def equalize_histogram_with_mask(image, mask):
    # Преобразуем изображение в LAB цветовое пространство
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Извлекаем L компоненту
    L_channel = lab_image[:, :, 0]

    # Сохраняем копию L для гистограммы "до"
    L_before = L_channel.copy()

    # Создаем маску для L компоненты
    masked_L = np.ma.masked_array(L_channel, mask=~mask)

    # Вычисляем гистограмму и кумулятивную распределительную функцию (CDF)
    hist, bins = np.histogram(masked_L.compressed(), bins=256, range=(0, 256))
    total_pixels = hist.sum()
    hist_percent = hist / total_pixels * 100
    cdf = hist.cumsum()

    # Нормализуем CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min()) # changed 255 to 100
    cdf_percent = cdf / cdf.max() * 100  # добавлено
    # cdf_normalized = np.ma.filled(cdf_normalized, 0).astype('uint8')
    cdf_map = np.ma.filled(cdf_normalized, 0).astype('uint8')  # изменено

    # Применяем преобразование к L компоненте
    # L_channel_equalized = cdf_normalized[L_channel]
    L_channel_equalized = cdf_map[L_channel]


    # Сохраняем копию L после эквализации
    L_after = L_channel_equalized.copy()

    # Обновляем L компоненту в LAB изображении
    lab_image[:, :, 0] = L_channel_equalized

    # Преобразуем LAB обратно в BGR
    equalized_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)

    # Гистограммы и CDF до/после
    hist_after, _ = np.histogram(L_after.ravel(), bins=256, range=(0, 256))
    total_pixels_after = hist_after.sum()
    hist_percent_after = hist_after / total_pixels_after * 100
    cdf_after = hist_after.cumsum()
    cdf_percent_after = cdf_after / cdf_after.max() * 100

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # До эквализации
    plt.subplot(1, 2, 1)
    plt.title('До эквализации')
    plt.hist(L_before.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.6, label='Гистограмма (%)')
    plt.plot(cdf / cdf.max() * hist.max(), color='red', label='CDF (%)')
    plt.xlabel('L значение')
    plt.ylabel('Проценты (%)')
    plt.legend()

    # После эквализации
    plt.subplot(1, 2, 2)
    plt.title('После эквализации')
    plt.hist(L_after.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.6, label='Гистограмма')
    plt.plot(cdf_after / cdf_after.max() * hist_after.max(), color='red', label='CDF (%)')
    plt.xlabel('L значение')
    plt.ylabel('Проценты (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return equalized_image

def create_mask(image_path, output_path, threshold=150):
    # Открываем изображение
    img = Image.open(image_path).convert('RGB')

    # Применяем фильтр размытия для уменьшения шума
    blurred = img.filter(ImageFilter.GaussianBlur(radius=2))

    # Преобразуем изображение в массив NumPy
    data = np.array(blurred)
    data1 = np.array(img)

    # Определяем каналы
    r, g, b = data.T

    # Создаем маску для белых областей
    mask = (r > threshold) & (g > threshold) & (b > threshold)

    return np.logical_not(mask.T)
    # return mask.T

# Пример использования
image_path = "rectified_object.jpg"
output_path = "lab_correction.jpg"


mask = create_mask(image_path, output_path)

# Загружаем изображение
image = cv2.imread(image_path)

# Создаем маску (например, круг в центре изображения)

# Выполняем эквализацию гистограммы с маской
equalized_image = equalize_histogram_with_mask(image, mask)

# Сохраняем результат
cv2.imwrite(output_path, equalized_image)
# cv2.imshow("window", equalized_image)
# cv2.waitKey()