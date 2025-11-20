"""
Детекция дефектов печатных плат:
1. Эквализация яркости цветов (после гомографии)
2. Детекция компонентов (проверка наличия, отсутствия, лишних)
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import cdist
from matplotlib.patches import Rectangle, Circle
from new_detect import (
    PCBCornerDetector,
    extract_pcb_region,
    adapt_homography_for_cropped_regions,
    transform_pcb_to_reference
)


def equalize_brightness(reference_image, test_image, method='histogram_matching'):
    """
    Эквализирует яркость тестового изображения относительно эталонного.
    
    Parameters:
    -----------
    reference_image : numpy.ndarray
        Эталонное изображение (после гомографии)
    test_image : numpy.ndarray
        Тестовое изображение (после гомографии)
    method : str
        Метод эквализации: 'histogram_matching' или 'mean_std'
    
    Returns:
    --------
    numpy.ndarray
        Тестовое изображение с эквализированной яркостью
    """
    # Нормализуем изображения в диапазон [0, 1]
    if reference_image.max() > 1.0:
        ref_norm = reference_image.astype(np.float32) / 255.0
    else:
        ref_norm = reference_image.astype(np.float32)
    
    if test_image.max() > 1.0:
        test_norm = test_image.astype(np.float32) / 255.0
    else:
        test_norm = test_image.astype(np.float32)
    
    if method == 'histogram_matching':
        return _histogram_matching(ref_norm, test_norm)
    elif method == 'mean_std':
        return _mean_std_matching(ref_norm, test_norm)
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def _histogram_matching(reference, test):
    """
    Сопоставление гистограмм для эквализации яркости.
    """
    # Работаем с grayscale для вычисления трансформации
    if len(reference.shape) == 3:
        ref_gray = np.dot(reference[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        ref_gray = reference.copy()
    
    if len(test.shape) == 3:
        test_gray = np.dot(test[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        test_gray = test.copy()
    
    # Вычисляем кумулятивные гистограммы
    bins = 256
    ref_hist, ref_bins = np.histogram(ref_gray.flatten(), bins=bins, range=(0, 1))
    test_hist, test_bins = np.histogram(test_gray.flatten(), bins=bins, range=(0, 1))
    
    # Нормализуем гистограммы
    ref_cdf = ref_hist.cumsum()
    ref_cdf = ref_cdf / ref_cdf[-1] if ref_cdf[-1] > 0 else ref_cdf
    
    test_cdf = test_hist.cumsum()
    test_cdf = test_cdf / test_cdf[-1] if test_cdf[-1] > 0 else test_cdf
    
    # Создаем lookup table для трансформации
    lookup = np.zeros(bins)
    for i in range(bins):
        # Находим значение в test_cdf, которое соответствует ref_cdf[i]
        idx = np.searchsorted(test_cdf, ref_cdf[i])
        lookup[i] = min(idx, bins - 1)
    
    # Применяем трансформацию к grayscale
    test_gray_quantized = (test_gray * (bins - 1)).astype(int)
    test_gray_quantized = np.clip(test_gray_quantized, 0, bins - 1)
    test_gray_equalized = lookup[test_gray_quantized] / (bins - 1)
    
    # Если изображение цветное, применяем трансформацию к каждому каналу
    if len(test.shape) == 3:
        # Вычисляем коэффициент масштабирования для каждого пикселя
        scale_factor = np.ones_like(test_gray)
        mask = test_gray > 0
        scale_factor[mask] = test_gray_equalized[mask] / (test_gray[mask] + 1e-8)
        
        # Применяем масштабирование к каждому каналу
        result = test.copy()
        for c in range(test.shape[2]):
            result[..., c] = np.clip(test[..., c] * scale_factor, 0, 1)
        return result
    else:
        return test_gray_equalized


def _mean_std_matching(reference, test):
    """
    Нормализация по среднему и стандартному отклонению.
    """
    # Работаем с grayscale для вычисления статистики
    if len(reference.shape) == 3:
        ref_gray = np.dot(reference[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        ref_gray = reference.copy()
    
    if len(test.shape) == 3:
        test_gray = np.dot(test[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        test_gray = test.copy()
    
    # Вычисляем статистику (игнорируем черные области)
    ref_valid = ref_gray[ref_gray > 0.1]
    test_valid = test_gray[test_gray > 0.1]
    
    if len(ref_valid) == 0 or len(test_valid) == 0:
        return test
    
    ref_mean = np.mean(ref_valid)
    ref_std = np.std(ref_valid) + 1e-8
    
    test_mean = np.mean(test_valid)
    test_std = np.std(test_valid) + 1e-8
    
    # Нормализуем тестовое изображение
    if len(test.shape) == 3:
        result = test.copy()
        for c in range(test.shape[2]):
            # Нормализуем: (x - mean_test) * (std_ref / std_test) + mean_ref
            result[..., c] = (test[..., c] - test_mean) * (ref_std / test_std) + ref_mean
        return np.clip(result, 0, 1)
    else:
        normalized = (test_gray - test_mean) * (ref_std / test_std) + ref_mean
        return np.clip(normalized, 0, 1)


class ComponentDetector:
    """
    Класс для детекции компонентов на печатной плате.
    """
    
    def __init__(self, min_component_size=50, max_component_size=50000):
        """
        Parameters:
        -----------
        min_component_size : int
            Минимальный размер компонента в пикселях
        max_component_size : int
            Максимальный размер компонента в пикселях
        """
        self.min_component_size = min_component_size
        self.max_component_size = max_component_size
    
    def detect_components(self, image, mask=None):
        """
        Обнаруживает компоненты на изображении платы.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Изображение платы (после гомографии и эквализации)
        mask : numpy.ndarray, optional
            Маска области платы
        
        Returns:
        --------
        list of dict
            Список компонентов, каждый содержит:
            - 'center': (x, y) - центр компонента
            - 'bbox': (x_min, y_min, x_max, y_max) - ограничивающий прямоугольник
            - 'area': int - площадь компонента
            - 'label': int - метка компонента
        """
        # Преобразуем в grayscale
        if len(image.shape) == 3:
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image.copy()
        
        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0
        
        # Применяем маску если есть
        if mask is not None:
            if mask.shape != gray.shape:
                mask = np.ones_like(gray, dtype=bool)
            gray = gray * mask.astype(float)
        
        # Адаптивная пороговая обработка для выделения компонентов
        # Компоненты обычно темнее или светлее фона
        valid_pixels = gray[gray > 0.1]  # игнорируем черные области
        if len(valid_pixels) == 0:
            return []
        
        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)
        
        # Два порога: для темных и светлых компонентов
        threshold_low = mean_val - 2 * std_val
        threshold_high = mean_val + 2 * std_val
        
        # Создаем бинарное изображение
        binary = np.zeros_like(gray, dtype=bool)
        binary[(gray < threshold_low) | (gray > threshold_high)] = True
        
        # Морфологическая обработка для очистки
        # Удаляем мелкий шум
        binary = ndimage.binary_opening(binary, structure=np.ones((3, 3)))
        # Заполняем небольшие дыры
        binary = ndimage.binary_closing(binary, structure=np.ones((5, 5)))
        
        # Применяем маску если есть
        if mask is not None:
            binary = binary & mask
        
        # Находим связные компоненты
        labeled, num_features = ndimage.label(binary)
        
        components = []
        for label_id in range(1, num_features + 1):
            # Извлекаем компонент
            component_mask = (labeled == label_id)
            area = np.sum(component_mask)
            
            # Фильтруем по размеру
            if area < self.min_component_size or area > self.max_component_size:
                continue
            
            # Вычисляем центр масс
            y_coords, x_coords = np.where(component_mask)
            center_x = float(np.mean(x_coords))
            center_y = float(np.mean(y_coords))
            
            # Вычисляем bounding box
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            
            components.append({
                'center': (center_x, center_y),
                'bbox': (x_min, y_min, x_max, y_max),
                'area': int(area),
                'label': label_id,
                'mask': component_mask
            })
        
        return components
    
    def match_components(self, reference_components, test_components, max_distance=50):
        """
        Сопоставляет компоненты между эталонным и тестовым изображениями.
        
        Parameters:
        -----------
        reference_components : list of dict
            Компоненты на эталонном изображении
        test_components : list of dict
            Компоненты на тестовом изображении
        max_distance : float
            Максимальное расстояние для сопоставления компонентов
        
        Returns:
        --------
        dict
            Результаты сопоставления:
            - 'matched': список пар (ref_idx, test_idx)
            - 'missing': список индексов отсутствующих компонентов из эталона
            - 'extra': список индексов лишних компонентов в тесте
        """
        if len(reference_components) == 0:
            return {
                'matched': [],
                'missing': [],
                'extra': list(range(len(test_components)))
            }
        
        if len(test_components) == 0:
            return {
                'matched': [],
                'missing': list(range(len(reference_components))),
                'extra': []
            }
        
        # Извлекаем центры компонентов
        ref_centers = np.array([comp['center'] for comp in reference_components])
        test_centers = np.array([comp['center'] for comp in test_components])
        
        # Вычисляем матрицу расстояний
        distances = cdist(ref_centers, test_centers)
        
        # Жадный алгоритм сопоставления (ближайший сосед)
        matched = []
        used_test = set()
        
        # Сортируем по расстоянию
        pairs = []
        for i in range(len(reference_components)):
            for j in range(len(test_components)):
                pairs.append((i, j, distances[i, j]))
        
        pairs.sort(key=lambda x: x[2])
        
        # Сопоставляем компоненты
        for ref_idx, test_idx, dist in pairs:
            if dist > max_distance:
                continue
            if test_idx in used_test:
                continue
            
            matched.append((ref_idx, test_idx))
            used_test.add(test_idx)
        
        # Находим отсутствующие компоненты
        matched_ref = set([m[0] for m in matched])
        missing = [i for i in range(len(reference_components)) if i not in matched_ref]
        
        # Находим лишние компоненты
        matched_test = set([m[1] for m in matched])
        extra = [i for i in range(len(test_components)) if i not in matched_test]
        
        return {
            'matched': matched,
            'missing': missing,
            'extra': extra
        }
    
    def visualize_components(self, image, components, matches=None, 
                            missing_indices=None, extra_indices=None, 
                            matched_indices=None,
                            title="Детекция компонентов"):
        """
        Визуализирует обнаруженные компоненты на изображении.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(image)
        
        for idx, comp in enumerate(components):
            x_min, y_min, x_max, y_max = comp['bbox']
            width = x_max - x_min
            height = y_max - y_min
            
            # Определяем цвет в зависимости от статуса
            if missing_indices and idx in missing_indices:
                color = 'red'
                label = 'Отсутствует'
            elif extra_indices and idx in extra_indices:
                color = 'orange'
                label = 'Лишний'
            elif matched_indices and idx in matched_indices:
                color = 'green'
                label = 'OK'
            elif matches:
                # Проверяем, сопоставлен ли компонент (для тестового изображения)
                is_matched = any(idx == m[1] for m in matches)
                if is_matched:
                    color = 'green'
                    label = 'OK'
                else:
                    color = 'yellow'
                    label = 'Не сопоставлен'
            else:
                color = 'blue'
                label = f'Comp {idx}'
            
            # Рисуем прямоугольник
            rect = Rectangle((x_min, y_min), width, height,
                           fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            # Рисуем центр
            center_x, center_y = comp['center']
            circle = Circle((center_x, center_y), 3, color=color, fill=True)
            ax.add_patch(circle)
            
            # Добавляем текст
            ax.text(x_min, y_min - 5, label, color=color, fontsize=8, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        return fig


def to_grayscale_float(image):
    """Преобразует изображение в float grayscale [0, 1]."""
    if len(image.shape) == 3:
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = image.astype(np.float32)

    if gray.max() > 1.0:
        gray = gray.astype(np.float32) / 255.0
    return gray


def compute_local_difference_map(reference_image,
                                 test_image,
                                 mask=None,
                                 window_size=11,
                                 stride=1,
                                 smooth_sigma=None):
    """
    Вычисляет карту локальных различий: для каждой точки усредняет (или усиливает)
    разницу в окне вокруг нее. Фактически имитируем "смотрим вокруг точки".
    """
    ref_gray = to_grayscale_float(reference_image)
    test_gray = to_grayscale_float(test_image)

    if ref_gray.shape != test_gray.shape:
        raise ValueError("Изображения для карты различий должны совпадать по размеру")

    diff = np.abs(ref_gray - test_gray)

    if mask is not None:
        if mask.shape != diff.shape:
            raise ValueError("Маска должна совпадать по размеру с изображениями")
        mask_float = mask.astype(np.float32)
        diff = diff * mask_float
    else:
        mask_float = np.ones_like(diff, dtype=np.float32)

    # Усредняем абсолютные разницы в окрестности каждой точки.
    local_sum = ndimage.uniform_filter(diff, size=window_size, mode='reflect')
    weights = ndimage.uniform_filter(mask_float, size=window_size, mode='reflect')
    weights = np.clip(weights, 1e-6, None)
    local_mean = local_sum / weights

    # Дополнительно считаем локальный максимум (чувствителен к "острым" дефектам).
    local_max = ndimage.maximum_filter(diff, size=window_size, mode='reflect')

    if smooth_sigma:
        local_mean = ndimage.gaussian_filter(local_mean, sigma=smooth_sigma)
        local_max = ndimage.gaussian_filter(local_max, sigma=smooth_sigma)

    if stride > 1:
        # Подвыборка (можно использовать для ускорения)
        local_mean = local_mean[::stride, ::stride]
        local_max = local_max[::stride, ::stride]
        # Возвращаем к исходному размеру простым масштабированием
        zoom_factor_y = diff.shape[0] / local_mean.shape[0]
        zoom_factor_x = diff.shape[1] / local_max.shape[1]
        local_mean = ndimage.zoom(local_mean, zoom=(zoom_factor_y, zoom_factor_x), order=1)
        local_max = ndimage.zoom(local_max, zoom=(zoom_factor_y, zoom_factor_x), order=1)

    return {
        "diff": diff,
        "mean": local_mean,
        "max": local_max
    }


def extract_difference_hotspots(local_map,
                                threshold=0.2,
                                min_area=25,
                                top_k=20):
    """
    Выделяет наиболее сильные области различий по карте.
    """
    if local_map.size == 0:
        return [], np.zeros_like(local_map)

    map_min = float(np.min(local_map))
    map_max = float(np.max(local_map))
    if map_max - map_min < 1e-8:
        normalized = np.zeros_like(local_map)
    else:
        normalized = (local_map - map_min) / (map_max - map_min)

    mask = normalized >= threshold
    labeled, num = ndimage.label(mask)

    hotspots = []
    for label_id in range(1, num + 1):
        region = labeled == label_id
        area = int(np.sum(region))
        if area < min_area:
            continue
        ys, xs = np.where(region)
        score = float(np.max(normalized[region]))
        hotspots.append({
            "center": (float(np.mean(xs)), float(np.mean(ys))),
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "area": area,
            "score": score
        })

    hotspots.sort(key=lambda h: h["score"], reverse=True)
    if top_k:
        hotspots = hotspots[:top_k]

    return hotspots, normalized


def visualize_difference_map(base_image,
                             normalized_map,
                             hotspots=None,
                             title="Карта локальных различий",
                             cmap="inferno",
                             alpha=0.6):
    """
    Визуализирует карту различий поверх изображения платы.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(base_image)
    heat = ax.imshow(normalized_map, cmap=cmap, alpha=alpha)
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04, label="Уровень различий")

    if hotspots:
        for idx, hotspot in enumerate(hotspots, start=1):
            x_min, y_min, x_max, y_max = hotspot["bbox"]
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="cyan",
                linewidth=2,
                linestyle="--"
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_min - 5,
                f"{idx}: {hotspot['score']:.2f}",
                color="cyan",
                fontsize=9,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.4)
            )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    return fig


def main():
    """
    Основная функция для детекции дефектов компонентов на печатных платах.
    """
    # Загрузка изображений
    reference_path = 'photoetalon.jpg'  # путь к эталонному изображению
    test_path = 'photoschakal.jpg'  # путь к тестовому изображению
    
    print("Загрузка изображений...")
    reference_img = Image.open(reference_path)
    test_img = Image.open(test_path)
    
    reference_array = np.array(reference_img)
    test_array = np.array(test_img)
    
    print(f"Эталон: {reference_array.shape}, Тест: {test_array.shape}")
    
    # Шаг 1: Находим углы и выравниваем изображения (гомография)
    print("\n=== ШАГ 1: ГОМОГРАФИЯ (выравнивание) ===")
    detector = PCBCornerDetector()
    H, aligned_corners, ref_corners, test_corners = detector.align_and_compare(
        reference_array, test_array
    )
    
    if H is None:
        print("Не удалось вычислить гомографию")
        return
    
    # Шаг 2: Вырезаем области плат
    print("\n=== ШАГ 2: ВЫРЕЗАНИЕ ОБЛАСТЕЙ ПЛАТ ===")
    reference_pcb_region, ref_bbox, ref_mask = extract_pcb_region(
        reference_array, ref_corners
    )
    test_pcb_region, test_bbox, test_mask = extract_pcb_region(
        test_array, test_corners
    )
    
    print(f"Вырезанная эталонная плата: {reference_pcb_region.shape}")
    print(f"Вырезанная тестовая плата: {test_pcb_region.shape}")
    
    # Шаг 3: Адаптируем гомографию для вырезанных областей
    H_adapted = adapt_homography_for_cropped_regions(H, ref_bbox, test_bbox)
    
    # Шаг 4: Преобразуем тестовую плату в координаты эталонной
    print("\n=== ШАГ 3: ПРЕОБРАЗОВАНИЕ ТЕСТОВОЙ ПЛАТЫ ===")
    transformed_test_pcb = transform_pcb_to_reference(
        test_pcb_region, H_adapted, reference_pcb_region.shape
    )
    transformed_test_mask = transform_pcb_to_reference(
        (test_mask.astype(np.uint8) * 255),
        H_adapted,
        reference_pcb_region.shape[:2]
    ) > 0
    
    # Шаг 5: ЭКВАЛИЗАЦИЯ ЯРКОСТИ (после гомографии)
    print("\n=== ШАГ 4: ЭКВАЛИЗАЦИЯ ЯРКОСТИ ===")
    transformed_test_pcb_equalized = equalize_brightness(
        reference_pcb_region,
        transformed_test_pcb,
        method='histogram_matching'  # или 'mean_std'
    )
    print("Эквализация яркости выполнена")
    
    # Шаг 5: КАРТА ЛОКАЛЬНЫХ РАЗЛИЧИЙ
    print("\n=== ШАГ 5: КАРТА ЛОКАЛЬНЫХ РАЗЛИЧИЙ ===")
    overlap_mask = ref_mask & transformed_test_mask
    diff_maps = compute_local_difference_map(
        reference_pcb_region,
        transformed_test_pcb_equalized,
        mask=overlap_mask,
        window_size=15,
        smooth_sigma=1.0
    )
    hotspots, normalized_heatmap = extract_difference_hotspots(
        diff_maps["mean"],
        threshold=0.2,
        min_area=20,
        top_k=15
    )

    print(f"Карта различий рассчитана. Найдено горячих зон: {len(hotspots)}")
    if hotspots:
        print("Топ зон по интенсивности:")
        for idx, hotspot in enumerate(hotspots[:5], start=1):
            print(f"  {idx}. Центр={hotspot['center']}, площадь={hotspot['area']} пикс., score={hotspot['score']:.3f}")

    # Визуализация карты различий
    heatmap_fig = visualize_difference_map(
        reference_pcb_region,
        normalized_heatmap,
        hotspots=hotspots,
        title="Карта локальных различий (усреднение по окрестностям)"
    )
    plt.show()

    # Шаг 6: ДЕТЕКЦИЯ КОМПОНЕНТОВ
    print("\n=== ШАГ 6: ДЕТЕКЦИЯ КОМПОНЕНТОВ ===")
    component_detector = ComponentDetector(
        min_component_size=50,      # минимальный размер компонента в пикселях
        max_component_size=50000,   # максимальный размер компонента в пикселях
    )
    
    print("Детекция компонентов на эталонном изображении...")
    ref_components = component_detector.detect_components(
        reference_pcb_region, 
        ref_mask
    )
    print(f"Найдено компонентов на эталоне: {len(ref_components)}")
    
    print("Детекция компонентов на тестовом изображении...")
    test_components = component_detector.detect_components(
        transformed_test_pcb_equalized,
        transformed_test_mask
    )
    print(f"Найдено компонентов на тесте: {len(test_components)}")
    
    # Шаг 7: Сопоставление компонентов
    print("\n=== ШАГ 7: СОПОСТАВЛЕНИЕ КОМПОНЕНТОВ ===")
    matches = component_detector.match_components(
        ref_components, 
        test_components, 
        max_distance=50  # максимальное расстояние для сопоставления
    )
    
    print(f"\n=== РЕЗУЛЬТАТЫ ДЕТЕКЦИИ КОМПОНЕНТОВ ===")
    print(f"Сопоставлено компонентов: {len(matches['matched'])}")
    print(f"Отсутствующих компонентов: {len(matches['missing'])}")
    print(f"Лишних компонентов: {len(matches['extra'])}")
    
    if matches['missing']:
        print("\nОтсутствующие компоненты:")
        for idx in matches['missing']:
            comp = ref_components[idx]
            print(f"  Компонент {idx}: центр {comp['center']}, площадь {comp['area']}")
    
    if matches['extra']:
        print("\nЛишние компоненты:")
        for idx in matches['extra']:
            comp = test_components[idx]
            print(f"  Компонент {idx}: центр {comp['center']}, площадь {comp['area']}")
    
    # Шаг 8: Визуализация результатов
    print("\n=== ШАГ 8: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ===")
    
    # Визуализация компонентов на эталонной плате
    fig_ref = component_detector.visualize_components(
        reference_pcb_region,
        ref_components,
        title="Компоненты на эталонной плате"
    )
    plt.show()
    
    # Визуализация компонентов на тестовой плате
    matched_test_indices = [m[1] for m in matches['matched']]
    fig_test = component_detector.visualize_components(
        transformed_test_pcb_equalized,
        test_components,
        matches=matches['matched'],
        matched_indices=matched_test_indices,
        extra_indices=matches['extra'],
        title="Компоненты на тестовой плате (зеленые = OK, оранжевые = лишние)"
    )
    plt.show()
    
    # Визуализация отсутствующих компонентов
    if matches['missing']:
        fig_missing = component_detector.visualize_components(
            reference_pcb_region,
            ref_components,
            missing_indices=matches['missing'],
            title="Отсутствующие компоненты на эталоне (красные)"
        )
        plt.show()
    
    # Дополнительная визуализация: сравнение изображений
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    axes[0, 0].imshow(reference_pcb_region)
    axes[0, 0].set_title('Эталонная плата')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(transformed_test_pcb)
    axes[0, 1].set_title('Тестовая плата (после гомографии)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(transformed_test_pcb_equalized)
    axes[1, 0].set_title('Тестовая плата (после эквализации)')
    axes[1, 0].axis('off')
    
    # Карту локальных различий показываем отдельно для наглядности
    axes[1, 1].imshow(reference_pcb_region)
    heat_overlay = axes[1, 1].imshow(normalized_heatmap, cmap='inferno', alpha=0.65)
    axes[1, 1].set_title('Локальные различия (heatmap)')
    axes[1, 1].axis('off')
    fig.colorbar(heat_overlay, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== ОБРАБОТКА ЗАВЕРШЕНА ===")


if __name__ == "__main__":
    main()

