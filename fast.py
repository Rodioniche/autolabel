import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def found_points(filename: str):
    # Загружаем в ЧЕРНО-БЕЛОЕ - это ключевое упрощение!
    img = Image.open(filename).convert('L')
    img_array = np.array(img)

    points = []

    # Просто ищем пиксели, которые сильно отличаются от соседей
    for i in range(1, img_array.shape[0] - 1):
        for j in range(1, img_array.shape[1] - 1):
            diff = (abs(int(img_array[i, j]) - int(img_array[i, j - 1])) +
                    abs(int(img_array[i, j]) - int(img_array[i, j + 1])) +
                    abs(int(img_array[i, j]) - int(img_array[i - 1, j])) +
                    abs(int(img_array[i, j]) - int(img_array[i + 1, j])))

            if diff > 120:
                points.append([i, j])

    return points, img_array


def make_picture(img_array, points):

    if len(img_array.shape) == 2:
        img_display = np.stack([img_array] * 3, axis=-1)
    else:
        img_display = img_array.copy()

    for y, x in points:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if 0 <= ny < img_display.shape[0] and 0 <= nx < img_display.shape[1]:
                    img_display[ny, nx] = [255, 0, 0]

    plt.figure(figsize=(12, 8))
    plt.imshow(img_display)
    plt.title(f'Найдено {len(points)} углов')
    plt.axis('off')
    plt.show()


filename = "платки.jpg"
points, img_array = found_points(filename)
make_picture(img_array, points)