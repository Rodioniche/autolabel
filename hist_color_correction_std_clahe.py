# ================================
#  отрисовка эквализации гистограммы и CLAHE
# ================================
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

# Загружаем изображение и переводим в LAB
image_path = "lab_correction.jpg"
image = cv2.imread(image_path)
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
L_channel = lab_image[:, :, 0]

# Вычисляем гистограмму
hist, bins = np.histogram(L_channel.flatten(), bins=256, range=(0, 256))

# Вычисляем CDF
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()  # нормируем для отрисовки на том же графике

image_path_CLAHE = "lab_correction_CLAHE.jpg"
image_CLAHE = cv2.imread(image_path_CLAHE)
lab_image_CLAHE = cv2.cvtColor(image_CLAHE, cv2.COLOR_BGR2Lab)
L_channel_CLAHE = lab_image_CLAHE[:, :, 0]

# Вычисляем гистограмму
hist_CLAHE, bins_CLAHE = np.histogram(L_channel_CLAHE.flatten(), bins=256, range=(0, 256))

# Вычисляем CDF
cdf_CLAHE = hist_CLAHE.cumsum()
cdf_normalized_CLAHE = cdf_CLAHE * hist_CLAHE.max() / cdf_CLAHE.max()  # нормируем для отрисовки на том же графике

# Построение графика
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.bar(range(256), hist, color='gray', alpha=0.6, label='Гистограмма L')
plt.plot(cdf_normalized, color='red', linewidth=2, label='CDF (кумулятивная функция)')
plt.xlabel('Значения L')
plt.ylabel('Количество пикселей')
plt.title('Гистограмма и CDF L-компоненты')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(range(256), hist_CLAHE, color='gray', alpha=0.6, label='Гистограмма L CLAHE')
plt.plot(cdf_normalized_CLAHE, color='red', linewidth=2, label='CDF CLAHE(кумулятивная функция)')
plt.xlabel('Значения L')
plt.ylabel('Количество пикселей')
plt.title('Гистограмма и CDF L-компоненты CLAHE')
plt.legend()
plt.show()