import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from matplotlib.path import Path
#ггйода гойдав

def find_equals(src_points, dst_points):
    """
    Вычисляет матрицу гомографии H (3x3) из соответствий точек
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError("Нужно минимум 4 точки для вычисления гомографии")

    if len(src_points) != len(dst_points):
        raise ValueError("Количество исходных и целевых точек должно совпадать")

    # Строим матрицу A для системы уравнений A * h = 0
    A = []
    for i in range(len(src_points)):
        x, y = src_points[i]
        x_prime, y_prime = dst_points[i]

        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y, -y_prime])

    A = np.array(A)

    # Решаем систему A * h = 0 методом SVD
    U, S, Vt = np.linalg.svd(A)

    # Последняя строка Vt соответствует наименьшему собственному значению
    H = Vt[-1].reshape(3, 3)

    # Нормализуем матрицу (делаем H[2,2] = 1)
    H = H / H[2, 2]
    print(f"H: {H}")

    return H


def extract_pcb_region(image, corners):
    """
    Вырезает область платы внутри углов

    Parameters:
    image: numpy array - исходное изображение
    corners: list - список углов [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

    Returns:
    numpy array - вырезанная область платы
    tuple - bounding box (min_x, min_y, max_x, max_y)
    """
    # Создаем путь по углам
    polygon_path = Path(corners)

    # Находим bounding box углов
    min_x = int(min(corners, key=lambda p: p[0])[0])
    max_x = int(max(corners, key=lambda p: p[0])[0])
    min_y = int(min(corners, key=lambda p: p[1])[1])
    max_y = int(max(corners, key=lambda p: p[1])[1])

    # Создаем маску для области внутри углов
    height, width = image.shape[:2]
    y_indices, x_indices = np.mgrid[min_y:max_y, min_x:max_x]
    points = np.column_stack([x_indices.ravel(), y_indices.ravel()])

    # Создаем маску точек внутри полигона
    mask = polygon_path.contains_points(points)
    mask = mask.reshape((max_y - min_y, max_x - min_x))

    # Создаем пустое изображение для вырезанной области
    if len(image.shape) == 3:
        extracted = np.zeros((max_y - min_y, max_x - min_x, image.shape[2]), dtype=image.dtype)
    else:
        extracted = np.zeros((max_y - min_y, max_x - min_x), dtype=image.dtype)

    # Копируем только пиксели внутри маски
    extracted[mask] = image[min_y:max_y, min_x:max_x][mask]

    return extracted, (min_x, min_y, max_x, max_y)


def transform_pcb_to_reference(test_pcb_region, H, reference_pcb_region_shape):
    """
    Преобразует вырезанную область тестовой платы в координаты вырезанной области эталонной платы

    Parameters:
    test_pcb_region: numpy array - вырезанная область тестовой платы
    H: numpy array (3x3) - матрица гомографии
    reference_pcb_region_shape: tuple - размер вырезанной области эталонной платы

    Returns:
    numpy array - преобразованная область тестовой платы
    """
    ref_height, ref_width = reference_pcb_region_shape[:2]

    # Создаем пустой массив того же размера, что и вырезанная эталонная область
    if len(reference_pcb_region_shape) == 3:
        transformed = np.zeros((ref_height, ref_width, reference_pcb_region_shape[2]), dtype=test_pcb_region.dtype)
    else:
        transformed = np.zeros((ref_height, ref_width), dtype=test_pcb_region.dtype)

    # Создаем координатную сетку для эталонной области
    y_ref, x_ref = np.indices((ref_height, ref_width))
    ones = np.ones_like(x_ref)

    # Преобразуем координаты в однородные
    ref_coords_homogeneous = np.stack([x_ref.ravel(), y_ref.ravel(), ones.ravel()])

    # Применяем обратную гомографию для получения координат в тестовом изображении
    H_inv = np.linalg.inv(H)
    test_coords_homogeneous = H_inv @ ref_coords_homogeneous

    # Нормализуем координаты
    test_x = test_coords_homogeneous[0] / test_coords_homogeneous[2]
    test_y = test_coords_homogeneous[1] / test_coords_homogeneous[2]

    # Округляем до целых пикселей
    test_x_int = np.round(test_x).astype(int)
    test_y_int = np.round(test_y).astype(int)

    # Создаем маску для валидных координат
    valid_mask = ((test_x_int >= 0) & (test_x_int < test_pcb_region.shape[1]) &
                  (test_y_int >= 0) & (test_y_int < test_pcb_region.shape[0]))

    # Преобразуем индексы обратно в 2D
    valid_indices_ref = np.where(valid_mask.reshape(ref_height, ref_width))
    valid_indices_test_y = test_y_int[valid_mask]
    valid_indices_test_x = test_x_int[valid_mask]

    # Копируем пиксели из тестовой области в эталонные координаты
    if len(test_pcb_region.shape) == 3:
        transformed[valid_indices_ref[0], valid_indices_ref[1]] = \
            test_pcb_region[valid_indices_test_y, valid_indices_test_x]
    else:
        transformed[valid_indices_ref[0], valid_indices_ref[1]] = \
            test_pcb_region[valid_indices_test_y, valid_indices_test_x]

    return transformed


def apply_homography(H, points):
    """
    Применяет матрицу гомографии H к точкам
    """
    points_array = np.array(points, dtype=np.float32)
    homogeneous_points = np.column_stack([points_array, np.ones(len(points_array))])

    transformed = (H @ homogeneous_points.T).T
    transformed_cartesian = transformed[:, :2] / transformed[:, 2:3]

    return [(float(x), float(y)) for x, y in transformed_cartesian]


class PCBCornerDetector:
    def __init__(self):
        pass

    def _bresenham_circle(self, radius):
        """
        Генерация точек окружности с помощью алгоритма Брезенхэма
        """
        x = 0
        y = radius
        d = 3 - 2 * radius
        points = set()

        def add_points(x, y):
            points.add((y, x))
            points.add((y, -x))
            points.add((-y, x))
            points.add((-y, -x))
            points.add((x, y))
            points.add((x, -y))
            points.add((-x, y))
            points.add((-x, -y))

        add_points(x, y)

        while y >= x:
            x += 1
            if d > 0:
                y -= 1
                d = d + 4 * (x - y) + 10
            else:
                d = d + 4 * x + 6
            add_points(x, y)

        return list(points)

    def find_pcb_corners_simple(self, image):
        """
        Простой метод поиска углов платы через анализ градиентов
        """
        if len(image.shape) == 3:
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image.copy()

        height, width = gray.shape
        edges = []

        # Поиск границ по всем сторонам
        border_size = min(height, width) // 10

        # Верхняя граница
        for j in range(border_size, width - border_size, 5):
            for i in range(border_size, height // 3):
                if abs(int(gray[i, j]) - int(gray[i - 1, j])) > 30:
                    edges.append((j, i))
                    break

        # Нижняя граница
        for j in range(border_size, width - border_size, 5):
            for i in range(height - border_size, height * 2 // 3, -1):
                if abs(int(gray[i, j]) - int(gray[i - 1, j])) > 30:
                    edges.append((j, i))
                    break

        # Левая граница
        for i in range(border_size, height - border_size, 5):
            for j in range(border_size, width // 3):
                if abs(int(gray[i, j]) - int(gray[i, j - 1])) > 30:
                    edges.append((j, i))
                    break

        # Правая граница
        for i in range(border_size, height - border_size, 5):
            for j in range(width - border_size, width * 2 // 3, -1):
                if abs(int(gray[i, j]) - int(gray[i, j - 1])) > 30:
                    edges.append((j, i))
                    break

        # Кластеризация по углам
        if len(edges) >= 4:
            corners = self._cluster_corners(edges, width, height)
            return self._order_corners(corners)

        # Fallback: углы изображения с отступом
        margin = 50
        return [
            [margin, margin],
            [width - margin, margin],
            [width - margin, height - margin],
            [margin, height - margin]
        ]

    def _cluster_corners(self, edges, width, height):
        """Кластеризует точки границ по углам"""
        quadrants = {
            'top_left': [], 'top_right': [],
            'bottom_left': [], 'bottom_right': []
        }

        center_x, center_y = width // 2, height // 2

        for x, y in edges:
            if x < center_x and y < center_y:
                quadrants['top_left'].append((x, y))
            elif x >= center_x and y < center_y:
                quadrants['top_right'].append((x, y))
            elif x < center_x and y >= center_y:
                quadrants['bottom_left'].append((x, y))
            else:
                quadrants['bottom_right'].append((x, y))

        corners = []
        for quadrant_name, points in quadrants.items():
            if points:
                if quadrant_name == 'top_left':
                    target = (0, 0)
                elif quadrant_name == 'top_right':
                    target = (width, 0)
                elif quadrant_name == 'bottom_left':
                    target = (0, height)
                else:
                    target = (width, height)

                closest = min(points, key=lambda p: math.sqrt((p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2))
                corners.append(closest)

        return corners

    def _order_corners(self, corners):
        """Упорядочивает углы"""
        if len(corners) != 4:
            return corners

        corners_sorted = sorted(corners, key=lambda p: p[1])
        top = sorted(corners_sorted[:2], key=lambda p: p[0])
        bottom = sorted(corners_sorted[2:], key=lambda p: p[0])

        return [top[0], top[1], bottom[1], bottom[0]]

    def align_and_compare(self, reference_image, test_image):
        """
        Находит углы на обоих изображениях, вычисляет гомографию и выравнивает

        Returns:
        tuple: (H_matrix, aligned_test_corners, reference_corners, test_corners)
        """
        # Находим углы на эталонном и тестовом изображениях
        reference_corners = self.find_pcb_corners_simple(reference_image)
        test_corners = self.find_pcb_corners_simple(test_image)

        print(f"Углы эталонного изображения: {reference_corners}")
        print(f"Углы тестового изображения: {test_corners}")

        if len(reference_corners) != 4 or len(test_corners) != 4:
            print("Не удалось найти 4 угла на одном из изображений")
            return None, test_corners, reference_corners, test_corners

        try:
            # Вычисляем матрицу гомографии
            H = find_equals(test_corners, reference_corners)
            print("Матрица гомографии H вычислена успешно")

            # Применяем гомографию к углам тестового изображения
            aligned_test_corners = apply_homography(H, test_corners)
            print(f"Выровненные углы тестового изображения: {aligned_test_corners}")

            return H, aligned_test_corners, reference_corners, test_corners

        except Exception as e:
            print(f"Ошибка при вычислении гомографии: {e}")
            return None, test_corners, reference_corners, test_corners

    def visualize_alignment(self, reference_image, test_image, H, aligned_corners, reference_corners, test_corners,
                            reference_pcb_region=None, test_pcb_region=None, transformed_test_pcb=None):
        """Визуализирует результаты выравнивания"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Эталонное изображение с углами
        axes[0, 0].imshow(reference_image)
        if reference_corners:
            x_ref = [p[0] for p in reference_corners]
            y_ref = [p[1] for p in reference_corners]
            axes[0, 0].plot(x_ref, y_ref, 'go', markersize=8, label='Эталон')
            for i in range(4):
                x1, y1 = reference_corners[i]
                x2, y2 = reference_corners[(i + 1) % 4]
                axes[0, 0].plot([x1, x2], [y1, y2], 'g-', linewidth=2)
        axes[0, 0].set_title('Эталонное изображение')
        axes[0, 0].legend()
        axes[0, 0].axis('off')

        # 2. Тестовое изображение с углами
        axes[0, 1].imshow(test_image)
        if test_corners:
            x_test = [p[0] for p in test_corners]
            y_test = [p[1] for p in test_corners]
            axes[0, 1].plot(x_test, y_test, 'ro', markersize=8, label='Исходные')
            for i in range(4):
                x1, y1 = test_corners[i]
                x2, y2 = test_corners[(i + 1) % 4]
                axes[0, 1].plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        axes[0, 1].set_title('Тестовое изображение')
        axes[0, 1].legend()
        axes[0, 1].axis('off')

        # 3. Вырезанные области плат
        if reference_pcb_region is not None and test_pcb_region is not None:
            axes[0, 2].imshow(reference_pcb_region)
            axes[0, 2].set_title('Вырезанная эталонная плата')
            axes[0, 2].axis('off')

            axes[1, 0].imshow(test_pcb_region)
            axes[1, 0].set_title('Вырезанная тестовая плата')
            axes[1, 0].axis('off')

            if transformed_test_pcb is not None:
                axes[1, 1].imshow(transformed_test_pcb)
                axes[1, 1].set_title('Преобразованная тестовая плата')
                axes[1, 1].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'Нет преобразованного изображения',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].axis('off')
        else:
            axes[0, 2].text(0.5, 0.5, 'Нет данных о вырезанных областях',
                            ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')

        # 6. Общая информация
        info_text = f"Эталон: {len(reference_corners)} углов\nТест: {len(test_corners)} углов"
        if H is not None:
            info_text += f"\nГомография: УСПЕХ"
            if reference_pcb_region is not None:
                info_text += f"\nРазмер эталонной платы: {reference_pcb_region.shape}"
            if test_pcb_region is not None:
                info_text += f"\nРазмер тестовой платы: {test_pcb_region.shape}"
        else:
            info_text += f"\nГомография: ОШИБКА"

        axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Статистика')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()


def display_transformed_image(transformed_array, title="Преобразованное изображение", figsize=(10, 8)):
    """
    Выводит изображение из преобразованного массива

    Parameters:
    transformed_array: numpy array - массив после преобразования
    title: str - заголовок изображения
    figsize: tuple - размер фигуры
    """
    plt.figure(figsize=figsize)

    # Проверяем тип данных массива
    if transformed_array.dtype == np.float32 or transformed_array.dtype == np.float64:
        # Если массив в float, нормализуем для отображения
        if transformed_array.max() > 1.0:
            display_array = transformed_array / 255.0
        else:
            display_array = transformed_array
    else:
        # Если массив в uint8, используем как есть
        display_array = transformed_array

    # Отображаем изображение
    if len(transformed_array.shape) == 3:
        plt.imshow(display_array)
    else:
        plt.imshow(display_array, cmap='gray')

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Дополнительная информация о массиве
    print(f"Размер массива: {transformed_array.shape}")
    print(f"Тип данных: {transformed_array.dtype}")
    print(f"Диапазон значений: {transformed_array.min()} - {transformed_array.max()}")
    print(f"Количество ненулевых пикселей: {np.count_nonzero(transformed_array)}")


def main():
    # Загрузка изображений
    reference_img = Image.open('chessboard_1_5_0.jpg')
    test_img = Image.open('chess.jpg')

    reference_array = np.array(reference_img)
    test_array = np.array(test_img)

    print(f"Эталон: {reference_array.shape}, Тест: {test_array.shape}")

    # Создаем детектор
    detector = PCBCornerDetector()

    # Выравниваем и сравниваем
    H, aligned_corners, ref_corners, test_corners = detector.align_and_compare(
        reference_array, test_array
    )

    if H is not None:
        # Вырезаем области плат из обоих изображений
        reference_pcb_region, ref_bbox = extract_pcb_region(reference_array, ref_corners)
        test_pcb_region, test_bbox = extract_pcb_region(test_array, test_corners)

        print(f"Вырезанная эталонная плата: {reference_pcb_region.shape}")
        print(f"Вырезанная тестовая плата: {test_pcb_region.shape}")

        # Преобразуем тестовую плату в координаты эталонной платы
        transformed_test_pcb = transform_pcb_to_reference(test_pcb_region, H, reference_pcb_region.shape)

        # Визуализируем результаты
        detector.visualize_alignment(
            reference_array, test_array, H, aligned_corners, ref_corners, test_corners,
            reference_pcb_region, test_pcb_region, transformed_test_pcb
        )

        # Показываем преобразованное изображение
        display_transformed_image(transformed_test_pcb, "Преобразованная тестовая плата")

    else:
        print("Не удалось вычислить гомографию, пропускаем преобразование")
        # Визуализируем результаты без преобразования
        detector.visualize_alignment(
            reference_array, test_array, H, aligned_corners, ref_corners, test_corners
        )


if __name__ == "__main__":
    main()