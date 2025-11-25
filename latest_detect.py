import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from matplotlib.path import Path


def get_homography(src_points, dst_points):
    """
    Вычисляет матрицу гомографии на основе четырех соответствующих точек.
    Из gomography.py - использует правильную формулу для растяжения изображений.

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
    print(f"H: {H}")

    return H


# Сохраняем старую функцию для обратной совместимости
def find_equals(src_points, dst_points):
    """Алиас для get_homography для обратной совместимости"""
    return get_homography(src_points, dst_points)


def extract_pcb_region(image, corners, padding=5):
    """
    Вырезает область платы внутри углов

    Parameters:
    image: numpy array - исходное изображение
    corners: list - список углов [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

    Returns:
    numpy array - вырезанная область платы
    tuple - bounding box (min_x, min_y, max_x, max_y)
    """
    if not corners or len(corners) < 4:
        h, w = image.shape[:2]
        return image.copy(), (0, 0, w, h), np.ones((h, w), dtype=bool)

    h, w = image.shape[:2]
    poly_path = Path(corners)
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    coords = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    mask = poly_path.contains_points(coords).reshape(h, w)

    if padding > 0:
        mask = ndimage.binary_dilation(mask, iterations=padding)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return image.copy(), (0, 0, w, h), mask

    min_y = max(0, ys.min())
    max_y = min(h, ys.max() + 1)
    min_x = max(0, xs.min())
    max_x = min(w, xs.max() + 1)

    clipped_mask = mask[min_y:max_y, min_x:max_x]

    if len(image.shape) == 3:
        masked_img = image.copy()
        masked_img[~mask] = 0
    else:
        masked_img = image.copy()
        masked_img[~mask] = 0

    extracted = masked_img[min_y:max_y, min_x:max_x]

    return extracted, (min_x, min_y, max_x, max_y), clipped_mask


def adapt_homography_for_cropped_regions(H, ref_bbox, test_bbox):
    """
    Адаптирует матрицу гомографии для работы с вырезанными областями

    Parameters:
    H: numpy array (3x3) - исходная матрица гомографии
    ref_bbox: tuple - bounding box эталонной области (min_x, min_y, max_x, max_y)
    test_bbox: tuple - bounding box тестовой области (min_x, min_y, max_x, max_y)

    Returns:
    numpy array - адаптированная матрица гомографии
    """
    ref_min_x, ref_min_y, ref_max_x, ref_max_y = ref_bbox
    test_min_x, test_min_y, test_max_x, test_max_y = test_bbox

    # Создаем матрицы смещения для эталонного и тестового изображений
    T_ref = np.array([
        [1, 0, -ref_min_x],
        [0, 1, -ref_min_y],
        [0, 0, 1]
    ])

    T_test_inv = np.array([
        [1, 0, test_min_x],
        [0, 1, test_min_y],
        [0, 0, 1]
    ])

    # Адаптируем матрицу гомографии: H_adapted = T_ref * H * T_test_inv
    H_adapted = T_ref @ H @ T_test_inv

    return H_adapted


def shift_corners_to_origin(corners, bbox):
    """
    Переводит координаты углов в локальную систему (0,0) относительно bounding-box.
    """
    if not corners or len(corners) < 4:
        return corners

    min_x, min_y = bbox[0], bbox[1]
    return [(x - min_x, y - min_y) for (x, y) in corners]


def subpixel(image, x, y):
    """
    Билинейная интерполяция для получения цвета пикселя в субпиксельных координатах.
    Из gomography.py - обеспечивает качественное растяжение изображений.
    """
    width_img, height_img = image.size
    x_A, y_A = int(round(x)), int(round(y))
    x_B, y_B = (x_A + 1), y_A
    x_C, y_C = x_A, (y_A + 1)
    x_D, y_D = (x_A + 1), (y_A + 1)

    if 0 <= x_A < width_img and 0 <= y_A < height_img and \
       0 <= x_B < width_img and 0 <= y_B < height_img and \
       0 <= x_C < width_img and 0 <= y_C < height_img and \
       0 <= x_D < width_img and 0 <= y_D < height_img:

        P_A = image.getpixel((x_A, y_A))
        P_B = image.getpixel((x_B, y_B))
        P_C = image.getpixel((x_C, y_C))
        P_D = image.getpixel((x_D, y_D))

        # Билинейная интерполяция по Y
        P_E = tuple(a * (1 - (y - y_A)) + b * (1 - (y_C - y)) for a, b in zip(P_A, P_C))
        P_F = tuple(a * (1 - (y - y_B)) + b * (1 - (y_D - y)) for a, b in zip(P_B, P_D))

        # Билинейная интерполяция по X
        P = tuple(int(a * (1 - (x - x_A)) + b * (1 - (x_B - x))) for a, b in zip(P_E, P_F))
    else:
        # Если координаты вне границ, возвращаем ближайший пиксель
        x_safe = max(0, min(width_img - 1, int(round(x))))
        y_safe = max(0, min(height_img - 1, int(round(y))))
        P = image.getpixel((x_safe, y_safe))

    return P


def apply_homography_img(image, H, width, height):
    """
    Применяет матрицу гомографии к изображению с билинейной интерполяцией.
    Из gomography.py - обеспечивает качественное растяжение изображений.

    :param image: Исходное изображение (PIL Image)
    :param H: Матрица гомографии 3x3
    :param width: Ширина выходного изображения
    :param height: Высота выходного изображения
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
            src_x, src_y = transformed_point[0], transformed_point[1]

            # Проверка, находятся ли новые координаты в пределах изображения
            if 0 <= src_x < width_img and 0 <= src_y < height_img:
                # Используем билинейную интерполяцию для качественного растяжения
                transformed_image.putpixel((x, y), subpixel(image, src_x, src_y))

    return transformed_image


def transform_pcb_to_reference(test_pcb_region, H_adapted, reference_pcb_region_shape, test_mask=None):
    """
    Преобразует вырезанную область тестовой платы в координаты вырезанной области эталонной платы.
    Теперь использует PIL с билинейной интерполяцией для качественного растяжения.
    Удаляет черный фон, оставляя только область платы.

    Parameters:
    test_pcb_region: numpy array - вырезанная область тестовой платы
    H_adapted: numpy array (3x3) - адаптированная матрица гомографии
    reference_pcb_region_shape: tuple - размер вырезанной области эталонной платы
    test_mask: numpy array (optional) - маска платы для удаления фона

    Returns:
    numpy array - преобразованная область тестовой платы без черного фона
    """
    ref_height, ref_width = reference_pcb_region_shape[:2]

    # Конвертируем numpy array в PIL Image для качественного преобразования
    if test_pcb_region.dtype != np.uint8:
        if test_pcb_region.max() <= 1.0:
            test_pcb_uint8 = (np.clip(test_pcb_region, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            test_pcb_uint8 = np.clip(test_pcb_region, 0, 255).astype(np.uint8)
    else:
        test_pcb_uint8 = test_pcb_region

    # Создаем PIL Image из numpy array
    if len(test_pcb_region.shape) == 3:
        test_pil = Image.fromarray(test_pcb_uint8, mode='RGB')
    else:
        # Grayscale -> RGB
        test_pil = Image.fromarray(test_pcb_uint8, mode='L').convert('RGB')

    # Применяем гомографию с билинейной интерполяцией
    transformed_pil = apply_homography_img(test_pil, H_adapted, ref_width, ref_height)

    # Конвертируем обратно в numpy array
    transformed_array = np.asarray(transformed_pil).astype(np.float32) / 255.0

    # Если исходное было grayscale, конвертируем обратно
    if len(test_pcb_region.shape) == 2:
        transformed_array = np.dot(transformed_array[..., :3], [0.2989, 0.5870, 0.1140])

    # Удаляем черный фон, используя маску или автоматическое определение
    if test_mask is not None:
        # Преобразуем маску так же, как изображение
        mask_uint8 = (test_mask.astype(np.uint8) * 255)
        if len(mask_uint8.shape) == 2:
            mask_pil = Image.fromarray(mask_uint8, mode='L').convert('RGB')
        else:
            mask_pil = Image.fromarray(mask_uint8, mode='RGB')
        
        transformed_mask_pil = apply_homography_img(mask_pil, H_adapted, ref_width, ref_height)
        transformed_mask = np.asarray(transformed_mask_pil).astype(np.float32) / 255.0
        
        # Берем только один канал маски
        if len(transformed_mask.shape) == 3:
            mask_gray = np.dot(transformed_mask[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            mask_gray = transformed_mask
        
        # Применяем маску: обнуляем пиксели вне платы (более строгий порог)
        mask_bool = mask_gray > 0.3  # Снижен порог для лучшего удаления фона
        
        # Обнуляем пиксели вне платы
        if len(transformed_array.shape) == 3:
            transformed_array[~mask_bool] = 0.0
        else:
            transformed_array[~mask_bool] = 0.0
    else:
        # Автоматическое определение черного фона
        # Черный фон - это пиксели с очень низкой яркостью
        if len(transformed_array.shape) == 3:
            brightness = np.mean(transformed_array, axis=2)
        else:
            brightness = transformed_array
        
        # Более умное определение фона: используем адаптивный порог
        # Фон - это пиксели, которые значительно темнее среднего значения
        mean_brightness = np.mean(brightness[brightness > 0.1])  # Среднее по не-черным пикселям
        if mean_brightness > 0:
            background_threshold = min(0.1, mean_brightness * 0.3)  # 30% от среднего или 0.1
        else:
            background_threshold = 0.05
        
        mask_bool = brightness > background_threshold
        
        # Обнуляем пиксели фона
        if len(transformed_array.shape) == 3:
            transformed_array[~mask_bool] = 0.0
        else:
            transformed_array[~mask_bool] = 0.0

    # Приводим к исходному типу данных
    if test_pcb_region.dtype == np.uint8:
        transformed_array = (transformed_array * 255.0).astype(np.uint8)
    else:
        transformed_array = transformed_array.astype(test_pcb_region.dtype)

    return transformed_array


def rectify_pcb_to_corners(image, corners, output_size=None):
    """
    Растягивает плату так, чтобы её углы совпадали с углами изображения.
    
    Parameters:
    image: numpy array - исходное изображение с платой
    corners: list - список углов платы [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    output_size: tuple (width, height) - размер выходного изображения. 
                 Если None, вычисляется автоматически на основе bounding box углов.
    
    Returns:
    numpy array - выпрямленное изображение платы
    numpy array - маска платы
    """
    if not corners or len(corners) < 4:
        return image, np.ones(image.shape[:2], dtype=bool)
    
    # Упорядочиваем углы: верх-левый, верх-правый, низ-правый, низ-левый
    corners_array = np.array(corners, dtype=np.float32)
    
    # Находим центр масс
    center = corners_array.mean(axis=0)
    
    # Сортируем углы по углу относительно центра
    angles = np.arctan2(corners_array[:, 1] - center[1], corners_array[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_corners = corners_array[sorted_indices]
    
    # Определяем верхние и нижние углы
    top = sorted_corners[sorted_corners[:, 1] < center[1]]
    bottom = sorted_corners[sorted_corners[:, 1] >= center[1]]
    
    if len(top) == 2 and len(bottom) == 2:
        # Сортируем верхние по x
        top = top[top[:, 0].argsort()]
        # Сортируем нижние по x
        bottom = bottom[bottom[:, 0].argsort()]
        # Порядок: верх-левый, верх-правый, низ-правый, низ-левый
        ordered_corners = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    else:
        ordered_corners = sorted_corners
    
    # Определяем размер выходного изображения
    if output_size is None:
        # Вычисляем размер на основе расстояний между углами
        width = int(max(
            np.linalg.norm(ordered_corners[1] - ordered_corners[0]),  # верхняя сторона
            np.linalg.norm(ordered_corners[2] - ordered_corners[3])   # нижняя сторона
        ))
        height = int(max(
            np.linalg.norm(ordered_corners[3] - ordered_corners[0]),  # левая сторона
            np.linalg.norm(ordered_corners[2] - ordered_corners[1])  # правая сторона
        ))
    else:
        width, height = output_size
    
    # Целевые углы - углы прямоугольного изображения
    dst_corners = np.array([
        [0, 0],           # верх-левый
        [width - 1, 0],   # верх-правый
        [width - 1, height - 1],  # низ-правый
        [0, height - 1]   # низ-левый
    ], dtype=np.float32)
    
    # Вычисляем гомографию для растяжения платы до углов изображения
    H = get_homography(ordered_corners, dst_corners)
    
    # Конвертируем numpy array в PIL Image
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    if len(image.shape) == 3:
        image_pil = Image.fromarray(image_uint8, mode='RGB')
    else:
        image_pil = Image.fromarray(image_uint8, mode='L').convert('RGB')
    
    # Применяем гомографию для растяжения платы
    rectified_pil = apply_homography_img(image_pil, H, width, height)
    
    # Конвертируем обратно в numpy array
    rectified_array = np.asarray(rectified_pil).astype(np.float32) / 255.0
    
    # Если исходное было grayscale, конвертируем обратно
    if len(image.shape) == 2:
        rectified_array = np.dot(rectified_array[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Создаем маску платы (вся область внутри прямоугольника)
    mask = np.ones((height, width), dtype=bool)
    
    # Приводим к исходному типу данных
    if image.dtype == np.uint8:
        rectified_array = (rectified_array * 255.0).astype(np.uint8)
    else:
        rectified_array = rectified_array.astype(image.dtype)
    
    return rectified_array, mask


def crop_to_pcb_region(image, threshold=0.05):
    """
    Обрезает изображение до минимального bounding box, содержащего плату (не черный фон).
    
    Parameters:
    image: numpy array - изображение с платой
    threshold: float - порог яркости для определения фона
    
    Returns:
    numpy array - обрезанное изображение
    tuple - (min_y, min_x, max_y, max_x) координаты обрезки
    """
    if len(image.shape) == 3:
        brightness = np.mean(image, axis=2)
    else:
        brightness = image
    
    # Находим пиксели платы (не фон)
    pcb_mask = brightness > threshold
    
    if not np.any(pcb_mask):
        # Если плата не найдена, возвращаем исходное изображение
        return image, (0, 0, image.shape[0], image.shape[1])
    
    # Находим bounding box
    rows = np.any(pcb_mask, axis=1)
    cols = np.any(pcb_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return image, (0, 0, image.shape[0], image.shape[1])
    
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    
    # Добавляем небольшой отступ
    padding = 5
    min_y = max(0, min_y - padding)
    min_x = max(0, min_x - padding)
    max_y = min(image.shape[0], max_y + padding + 1)
    max_x = min(image.shape[1], max_x + padding + 1)
    
    # Обрезаем изображение
    if len(image.shape) == 3:
        cropped = image[min_y:max_y, min_x:max_x, :]
    else:
        cropped = image[min_y:max_y, min_x:max_x]

    return cropped, (min_y, min_x, max_y, max_x)


def to_grayscale_float(image):
    """Переводит изображение в grayscale и нормализует в диапазон [0, 1]."""
    if len(image.shape) == 3:
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = image.astype(np.float32)

    if gray.max() > 1.0:
        gray = gray.astype(np.float32) / 255.0
    return gray


def compare_pcb_regions(reference_region, transformed_region,
                        reference_mask, transformed_mask,
                        diff_threshold=0.1):
    """
    Сравнивает две выровненные области плат и возвращает статистику и карту различий.
    """
    ref_gray = to_grayscale_float(reference_region)
    test_gray = to_grayscale_float(transformed_region)

    if reference_mask.shape != ref_gray.shape:
        reference_mask = np.ones_like(ref_gray, dtype=bool)
    else:
        reference_mask = reference_mask.astype(bool)

    if transformed_mask.shape != test_gray.shape:
        transformed_mask = np.ones_like(test_gray, dtype=bool)
    else:
        transformed_mask = transformed_mask.astype(bool)

    overlap_mask = reference_mask & transformed_mask
    if np.count_nonzero(overlap_mask) == 0:
        print("Нет пересечения масок для сравнения.")
        return None, None, None

    diff_map = np.abs(ref_gray - test_gray)
    diff_masked = diff_map[overlap_mask]

    mean_diff = float(np.mean(diff_masked))
    max_diff = float(np.max(diff_masked))
    high_diff_ratio = float(np.mean(diff_masked > diff_threshold))

    print("\n=== СРАВНЕНИЕ ПЛАТ ===")
    print(f"Средняя разница: {mean_diff:.4f}")
    print(f"Макс. разница: {max_diff:.4f}")
    print(f"Доля пикселей > {diff_threshold:.2f}: {high_diff_ratio*100:.2f}%")

    diff_visual = np.zeros_like(ref_gray)
    diff_visual[overlap_mask] = diff_map[overlap_mask]

    stats = {
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "high_diff_ratio": high_diff_ratio,
        "threshold": diff_threshold,
    }

    return stats, diff_visual, overlap_mask


def compare_pcb_by_grid(reference_region, transformed_region,
                        reference_mask, transformed_mask,
                        grid_size=8, diff_threshold=0.15):
    """
    Сравнивает платы по сетке прямоугольников, чтобы игнорировать локальные пиксельные сдвиги.
    """
    ref_gray = to_grayscale_float(reference_region)
    test_gray = to_grayscale_float(transformed_region)

    if reference_mask.shape != ref_gray.shape:
        reference_mask = np.ones_like(ref_gray, dtype=bool)
    else:
        reference_mask = reference_mask.astype(bool)

    if transformed_mask.shape != test_gray.shape:
        transformed_mask = np.ones_like(test_gray, dtype=bool)
    else:
        transformed_mask = transformed_mask.astype(bool)

    overlap_mask = reference_mask & transformed_mask
    if np.count_nonzero(overlap_mask) == 0:
        print("Нет пересечения масок для сравнения по сетке.")
        return None

    diff_map = np.abs(ref_gray - test_gray)

    height, width = diff_map.shape
    y_bounds = np.linspace(0, height, grid_size + 1, dtype=int)
    x_bounds = np.linspace(0, width, grid_size + 1, dtype=int)

    heatmap = np.full((grid_size, grid_size), np.nan, dtype=float)
    results = []

    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = y_bounds[i], y_bounds[i + 1]
            x0, x1 = x_bounds[j], x_bounds[j + 1]

            cell_mask = overlap_mask[y0:y1, x0:x1]
            valid_pixels = np.count_nonzero(cell_mask)
            if valid_pixels < 10:
                continue

            cell_diff = diff_map[y0:y1, x0:x1][cell_mask]
            mean_diff = float(np.mean(cell_diff))
            max_diff = float(np.max(cell_diff))

            heatmap[i, j] = mean_diff
            results.append({
                "grid_pos": (i, j),
                "coords": (x0, y0, x1, y1),
                "mean_diff": mean_diff,
                "max_diff": max_diff,
                "is_defect": mean_diff > diff_threshold,
            })

    if not results:
        print("Недостаточно информации для сравнения по сетке.")
        return None

    defect_count = sum(r["is_defect"] for r in results)
    print("\n=== СРАВНЕНИЕ ПО СЕТКЕ ===")
    print(f"Сегментов с данными: {len(results)} / {grid_size * grid_size}")
    print(f"Дефектных сегментов (> {diff_threshold:.2f}): {defect_count}")

    return {
        "results": results,
        "heatmap": heatmap,
        "grid_size": grid_size,
        "threshold": diff_threshold,
    }


def plot_corners(ax, corners, color='r'):
    if not corners:
        return
    xs = [float(p[0]) for p in corners]
    ys = [float(p[1]) for p in corners]
    xs_loop = xs + [xs[0]]
    ys_loop = ys + [ys[0]]
    ax.plot(xs_loop, ys_loop, color+'-', linewidth=2)
    ax.scatter(xs, ys, c=color, s=20)


def show_combined_visualization(reference_array,
                                test_array,
                                reference_pcb_region,
                                test_pcb_region,
                                transformed_test_pcb,
                                diff_visual,
                                ref_corners,
                                test_corners,
                                ref_corners_local,
                                test_corners_local,
                                grid_data):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(reference_array)
    plot_corners(axes[0, 0], ref_corners, 'g')
    axes[0, 0].set_title("Эталон (полный)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(test_array)
    plot_corners(axes[0, 1], test_corners, 'r')
    axes[0, 1].set_title("Тест (полный)")
    axes[0, 1].axis('off')

    if diff_visual is not None:
        im = axes[0, 2].imshow(diff_visual, cmap='inferno')
        axes[0, 2].set_title("Карта различий")
        axes[0, 2].axis('off')
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    else:
        axes[0, 2].axis('off')

    axes[1, 0].imshow(reference_pcb_region)
    plot_corners(axes[1, 0], ref_corners_local, 'g')
    axes[1, 0].set_title("Эталонная плата")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(transformed_test_pcb)
    axes[1, 1].set_title("Преобразованная плата")
    axes[1, 1].axis('off')

    if grid_data:
        axes[1, 2].imshow(reference_pcb_region)
        for res in grid_data["results"]:
            x0, y0, x1, y1 = res["coords"]
            color = 'red' if res["is_defect"] else 'lime'
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 fill=False, edgecolor=color, linewidth=2, alpha=0.8)
            axes[1, 2].add_patch(rect)
            axes[1, 2].text(x0 + 4, y0 + 12, f"{res['mean_diff']:.2f}",
                            color=color, fontsize=8, weight='bold')
        axes[1, 2].set_title("Сегменты (красные = > порога)")
        axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


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


def main():
    # Загрузка изображений
    reference_img = Image.open('chess.jpg')
    test_img = Image.open('chessboard_3_3.jpg')

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
        print("\n=== Растяжение плат до углов изображения ===")
        
        # Растягиваем обе платы так, чтобы их углы совпадали с углами изображения
        # Используем одинаковый размер для обеих плат (берем максимальный)
        ref_rectified, ref_mask_rectified = rectify_pcb_to_corners(reference_array, ref_corners)
        test_rectified, test_mask_rectified = rectify_pcb_to_corners(test_array, test_corners)
        
        # Используем размер эталонной платы для обеих
        ref_height, ref_width = ref_rectified.shape[:2]
        test_height, test_width = test_rectified.shape[:2]
        
        # Используем максимальный размер для обеих плат
        target_width = max(ref_width, test_width)
        target_height = max(ref_height, test_height)
        
        print(f"Размер эталонной платы после растяжения: {ref_rectified.shape}")
        print(f"Размер тестовой платы после растяжения: {test_rectified.shape}")
        print(f"Целевой размер для обеих плат: ({target_height}, {target_width})")
        
        # Перерастягиваем обе платы до одинакового размера
        if ref_rectified.shape[:2] != (target_height, target_width):
            ref_rectified, ref_mask_rectified = rectify_pcb_to_corners(
                reference_array, ref_corners, output_size=(target_width, target_height)
            )
        
        if test_rectified.shape[:2] != (target_height, target_width):
            test_rectified, test_mask_rectified = rectify_pcb_to_corners(
                test_array, test_corners, output_size=(target_width, target_height)
            )
        
        # Теперь обе платы растянуты до углов изображения и имеют одинаковый размер
        reference_pcb_region = ref_rectified
        transformed_test_pcb = test_rectified
        ref_mask = ref_mask_rectified
        transformed_test_mask = test_mask_rectified
        
        print(f"Финальный размер обеих плат: {reference_pcb_region.shape}")
        print("[i] Платы растянуты до углов изображения и готовы к сравнению")

        # Визуализируем результаты
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Исходные изображения с углами
        axes[0, 0].imshow(reference_array)
        x_ref = [p[0] for p in ref_corners]
        y_ref = [p[1] for p in ref_corners]
        axes[0, 0].plot(x_ref, y_ref, 'ro-', markersize=8)
        axes[0, 0].set_title('Эталон с углами')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(test_array)
        x_test = [p[0] for p in test_corners]
        y_test = [p[1] for p in test_corners]
        axes[0, 1].plot(x_test, y_test, 'ro-', markersize=8)
        axes[0, 1].set_title('Тест с углами')
        axes[0, 1].axis('off')

        # 2. Растянутые платы (углы платы = углы изображения)
        axes[1, 0].imshow(reference_pcb_region)
        # Показываем углы изображения (которые теперь совпадают с углами платы)
        h, w = reference_pcb_region.shape[:2]
        corners_img = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]
        x_corners = [p[0] for p in corners_img]
        y_corners = [p[1] for p in corners_img]
        axes[1, 0].plot(x_corners + [x_corners[0]], y_corners + [y_corners[0]], 'go-', markersize=6, linewidth=2)
        axes[1, 0].set_title(f'Эталонная плата (растянута)\n{reference_pcb_region.shape}')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(transformed_test_pcb)
        # Показываем углы изображения
        h_t, w_t = transformed_test_pcb.shape[:2]
        corners_img_t = [(0, 0), (w_t-1, 0), (w_t-1, h_t-1), (0, h_t-1)]
        x_corners_t = [p[0] for p in corners_img_t]
        y_corners_t = [p[1] for p in corners_img_t]
        axes[1, 1].plot(x_corners_t + [x_corners_t[0]], y_corners_t + [y_corners_t[0]], 'bo-', markersize=6, linewidth=2)
        axes[1, 1].set_title(f'Тестовая плата (растянута)\n{transformed_test_pcb.shape}')
        axes[1, 1].axis('off')

        # 3. Преобразованная плата
        axes[1, 2].imshow(transformed_test_pcb)
        axes[1, 2].set_title(f'Преобразованная\n{transformed_test_pcb.shape}')
        axes[1, 2].axis('off')

        # 4. Сравнение эталонной и преобразованной
        axes[0, 2].imshow(reference_pcb_region)
        axes[0, 2].set_title('Эталонная для сравнения')
        axes[0, 2].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Преобразованная тестовая плата: {transformed_test_pcb.shape}")
        print(f"Ненулевых пикселей: {np.count_nonzero(transformed_test_pcb)}")

        stats_pix, diff_visual, overlap_mask = compare_pcb_regions(
            reference_pcb_region,
            transformed_test_pcb,
            ref_mask,
            transformed_test_mask,
            diff_threshold=0.1
        )

        grid_info = compare_pcb_by_grid(
            reference_pcb_region,
            transformed_test_pcb,
            ref_mask,
            transformed_test_mask,
            grid_size=8,
            diff_threshold=0.13
        )

        # Создаем локальные углы для визуализации (углы изображения = углы платы после растяжения)
        h, w = reference_pcb_region.shape[:2]
        ref_corners_local = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]
        test_corners_local = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]
        
        show_combined_visualization(
            reference_array,
            test_array,
            reference_pcb_region,
            transformed_test_pcb,  # Используем растянутую тестовую плату
            transformed_test_pcb,
            diff_visual,
            ref_corners,
            test_corners,
            ref_corners_local,
            test_corners_local,
            grid_info
        )

    else:
        print("Не удалось вычислить гомографию")


if __name__ == "__main__":
    main()