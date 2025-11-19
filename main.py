import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def found_points(filename: str):
    img = Image.open(filename).convert('L')
    img_array = np.array(img)
    points = []
    count = 0
    mean_brightness = np.mean(img_array)
    dark_bright = mean_brightness - 30
    light_bright = mean_brightness + 30
    for i in range(2, img_array.shape[0] - 2):
        for j in range(2, img_array.shape[1] - 2):

            for k in range(i - 2, i + 2, 1):
                for l in range(j + 2, j - 2, -1):
                    if img_array[i][j][0] < 150 and img_array[i][j][1] < 150:
                        if img_array[k][l][0] < 150 and img_array[k][l][1] < 150:
                            count += 1

            if 16 <= count <= 32:
                points.append([i, j])
            count = 0
    return points, img_array


def make_picture(img_array, points):
    img_with_points = img_array.copy()

    for y, x in points:
        if 0 <= y < img_with_points.shape[0] and 0 <= x < img_with_points.shape[1]:
            img_with_points[y, x] = [255, 0, 0, 255]

    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_points)
    plt.title(f'Изображение с {len(points)} красными точками')
    plt.axis('off')
    plt.show()

filename = "mira.png"
points, img_array = found_points(filename)
make_picture(img_array, points)













