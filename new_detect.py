import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from matplotlib.path import Path


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


def transform_pcb_to_reference(test_pcb_region, H_adapted, reference_pcb_region_shape):
    """
    Преобразует вырезанную область тестовой платы в координаты вырезанной области эталонной платы

    Parameters:
    test_pcb_region: numpy array - вырезанная область тестовой платы
    H_adapted: numpy array (3x3) - адаптированная матрица гомографии
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

    # Применяем обратную адаптированную гомографию
    H_adapted_inv = np.linalg.inv(H_adapted)
    test_coords_homogeneous = H_adapted_inv @ ref_coords_homogeneous

    # Нормализуем координаты
    test_x = test_coords_homogeneous[0] / test_coords_homogeneous[2]
    test_y = test_coords_homogeneous[1] / test_coords_homogeneous[2]

    # Округляем до целых пикселей (ближайший сосед)
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
    reference_img = Image.open('photoetalon.jpg')
    test_img = Image.open('photoschakal.jpg')

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
        reference_pcb_region, ref_bbox, ref_mask = extract_pcb_region(reference_array, ref_corners)
        test_pcb_region, test_bbox, test_mask = extract_pcb_region(test_array, test_corners)

        # Нормализуем углы так, чтобы левый верхний угол стал (0,0)
        ref_corners_local = shift_corners_to_origin(ref_corners, ref_bbox)
        test_corners_local = shift_corners_to_origin(test_corners, test_bbox)
        aligned_corners_local = shift_corners_to_origin(aligned_corners, ref_bbox)

        print(f"Вырезанная эталонная плата: {reference_pcb_region.shape}")
        print(f"Вырезанная тестовая плата: {test_pcb_region.shape}")
        print(f"BBox эталон: {ref_bbox}")
        print(f"BBox тест: {test_bbox}")
        print(f"Локальные углы эталона: {ref_corners_local}")
        print(f"Локальные углы теста: {test_corners_local}")
        print(f"Локальные выровненные углы: {aligned_corners_local}")

        # Адаптируем матрицу гомографии для вырезанных областей
        H_adapted = adapt_homography_for_cropped_regions(H, ref_bbox, test_bbox)

        # Преобразуем тестовую плату в координаты эталонной платы
        transformed_test_pcb = transform_pcb_to_reference(test_pcb_region, H_adapted, reference_pcb_region.shape)
        transformed_test_mask = transform_pcb_to_reference(
            (test_mask.astype(np.uint8) * 255),
            H_adapted,
            reference_pcb_region.shape[:2]
        ) > 0

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

        # 2. Вырезанные области
        axes[1, 0].imshow(reference_pcb_region)
        if ref_corners_local:
            x_loc = [p[0] for p in ref_corners_local]
            y_loc = [p[1] for p in ref_corners_local]
            axes[1, 0].plot(x_loc, y_loc, 'go-', markersize=6)
        axes[1, 0].set_title(f'Эталонная плата\n{reference_pcb_region.shape}')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(test_pcb_region)
        if test_corners_local:
            x_loc_t = [p[0] for p in test_corners_local]
            y_loc_t = [p[1] for p in test_corners_local]
            axes[1, 1].plot(x_loc_t, y_loc_t, 'bo-', markersize=6)
        axes[1, 1].set_title(f'Тестовая плата\n{test_pcb_region.shape}')
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

        show_combined_visualization(
            reference_array,
            test_array,
            reference_pcb_region,
            test_pcb_region,
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