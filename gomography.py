# ================================
#  Преобразование
# ================================
import numpy as np
from PIL import Image

out_w, out_h = 1920, 1080

# pts_src = np.float32([[58, 21], [141, 35], [58, 75], [141, 89]])
# pts_dst = np.float32([[32, 32], [115, 32], [32, 86], [115, 86]])
pts_src = np.float32([[215, 177], [709, 190], [754, 522], [197, 531]])
pts_dst = np.float32([[0, 0], [out_w - 1, 0], [out_w-1, out_h-1], [0, out_h-1]])

input_path = '5253805435188350320.jpg'
output_path = 'rectified_object.jpg'

def get_homography(src_points, dst_points):
    """
    Вычисляет матрицу гомографии на основе четырех соответствующих точек.

    :param src_points: Список исходных точек (x, y)
    :param dst_points: Список целевых точек (x, y)
    :return: Матрица гомографии 3x3
    """
    assert len(src_points) == 4 and len(dst_points) == 4, "Должно быть ровно 4 точки."

    # Создание матрицы A
    A = []
    for i in range(4):
        x, y = src_points[i]
        x_prime, y_prime = dst_points[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    A = np.array(A)

    # Решение для матрицы H с использованием SVD
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Нормализация матрицы H
    H /= H[2, 2]

    return H

def subpixel(image, x, y):
  width_img, height_img = image.size
  x_A, y_A = int(round(x)), int(round(y))
  x_B, y_B = (x_A + 1), y_A
  x_C, y_C = x_A, (y_A + 1)
  x_D, y_D = (x_A + 1), (y_A + 1)

  if 0 <= x_A < width_img and 0 <= y_A < height_img and \
     0 <= x_B < width_img and 0 <= y_B < height_img and \
     0 <= x_C < width_img and 0 <= y_C < height_img and \
     0 <= x_D < width_img and 0 <= y_D < height_img:

    #  print(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D)
    #  print("here")
     P_A = image.getpixel((x_A, y_A))
    #  print("here1")
     P_B = image.getpixel((x_B, y_B))
    #  print("here2")
     P_C = image.getpixel((x_C, y_C))
    #  print("here3")
     P_D = image.getpixel((x_D, y_D))
    #  print("here4")


     P_E = tuple(a * (1 - (y - y_A)) + b * (1 - (y_C - y)) for a, b in zip(P_A, P_C))
     P_F = tuple(a * (1 - (y - y_B)) + b * (1 - (y_D - y)) for a, b in zip(P_B, P_D))
    #  P_E = P_A * (1 - (y - y_A)) + P_C * (1 - (y_C - y))
    #  P_F = P_B * (1 - (y - y_B)) + P_D * (1 - (y_D - y))


     P = tuple(int(a * (1 - (x - x_A)) + b * (1 - (x_B - x))) for a, b in zip(P_E, P_F))
    #  P = P_E * (1 - (x - x_A)) + P_F * (1 - (x_B - x))
  else:
    # print("catch 1")
    # print(x_A, y_A)
    P = image.getpixel((x_A, y_A))
    # print("catch 2")

  # print(P)
  return P


def apply_homography_img (image, H, width, height):
    """
    Применяет матрицу гомографии к изображению.

    :param image: Исходное изображение (PIL Image)
    :param H: Матрица гомографии 3x3
    :return: Преобразованное изображение (PIL Image)
    """
    # Получение размеров изображения
    width_img, height_img = image.size

    # Создание нового изображения для хранения результата
    transformed_image = Image.new("RGB", (width, height))

    # Преобразование матрицы гомографии в обратную
    H_inv = np.linalg.inv(H)

    for y in range(height):
        for x in range(width):
            # Преобразование координат пикселя в однородные координаты
            original_point = np.array([x, y, 1])
            transformed_point = H_inv @ original_point

            # Нормализация
            transformed_point /= transformed_point[2]

            # Получение новых координат
            # new_x, new_y = int(transformed_point[0]), int(transformed_point[1])
            new_x, new_y = int(round(transformed_point[0])), int(round(transformed_point[1]))

            # Проверка, находятся ли новые координаты в пределах изображения
            if 0 <= new_x < width_img and 0 <= new_y < height_img:
                # transformed_image.putpixel((x, y), image.getpixel((new_x, new_y)))
                transformed_image.putpixel((x, y), subpixel(image, transformed_point[0], transformed_point[1]))
                # print(image.getpixel((new_x, new_y)))

    return transformed_image

image = Image.open(input_path)
H = get_homography(pts_src, pts_dst)
transformed_image = apply_homography_img(image, H, width=out_w, height=out_h)

# Сохранение преобразованного изображения
transformed_image.save(output_path)