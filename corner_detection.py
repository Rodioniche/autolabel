import dataclasses
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    from skimage import color, filters, measure, morphology
    from skimage.filters import threshold_otsu
except ImportError:
    raise ImportError("Требуется scikit-image: pip install scikit-image")

Array = np.ndarray


def load_image(path: str | Path) -> Array:
    with Image.open(path) as img:
        img = img.convert("RGB")
        data = np.asarray(img).astype(np.float32) / 255.0
    return data


def to_grayscale(image: Array) -> Array:
    if image.ndim == 2:
        return image
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


def normalize_intensity(reference: Array, test: Array, mask: Array | None = None) -> Array:
    if mask is None:
        ref_vals = reference
        test_vals = test
    else:
        ref_vals = reference[mask]
        test_vals = test[mask]

    ref_mean, ref_std = ref_vals.mean(), ref_vals.std() + 1e-6
    test_mean, test_std = test_vals.mean(), test_vals.std() + 1e-6

    aligned = (test - test_mean) / test_std
    aligned = aligned * ref_std + ref_mean
    return np.clip(aligned, 0.0, 1.0)


def match_histogram(reference: Array, test: Array, mask: Array | None = None, bins: int = 256) -> Array:
    if mask is None:
        ref_vals = reference.ravel()
        test_vals = test.ravel()
    else:
        ref_vals = reference[mask]
        test_vals = test[mask]

    ref_hist, ref_bins = np.histogram(ref_vals, bins=bins, range=(0.0, 1.0), density=True)
    test_hist, _ = np.histogram(test_vals, bins=bins, range=(0.0, 1.0), density=True)

    ref_cdf = np.cumsum(ref_hist)
    test_cdf = np.cumsum(test_hist)
    ref_cdf /= ref_cdf[-1] if ref_cdf[-1] != 0 else 1.0
    test_cdf /= test_cdf[-1] if test_cdf[-1] != 0 else 1.0

    ref_centers = (ref_bins[:-1] + ref_bins[1:]) / 2.0
    mapping = np.interp(test_cdf, ref_cdf, ref_centers)

    indices = np.clip((test * (bins - 1)).astype(int), 0, bins - 1)
    matched = mapping[indices]
    return np.clip(matched, 0.0, 1.0)


def find_board_corners(
    image: Array,
    tolerance: float = 2.5,
    gaussian_sigma: float = 1.0,
    closing_radius: int = 3,
    hole_area_threshold: int = 5000,
    debug: bool = False,
) -> Optional[np.ndarray]:
    """
    Находит углы платы на изображении.
    
    Parameters:
    -----------
    image : Array
        Изображение в формате [0, 1], RGB или grayscale
    tolerance : float
        Точность аппроксимации многоугольника (меньше = больше точек)
    gaussian_sigma : float
        Сигма для размытия перед бинаризацией
    closing_radius : int
        Радиус диска для морфологического закрытия
    hole_area_threshold : int
        Минимальная площадь дырки для удаления
    debug : bool
        Показывать промежуточные результаты
    
    Returns:
    --------
    Optional[np.ndarray]
        Массив углов формы (N, 2) с координатами (y, x), или None если не найдено
    """
    # Конвертируем в uint8 для skimage (если нужно)
    if image.max() <= 1.0:
        image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    
    # RGB -> grayscale через skimage
    if image_uint8.ndim == 3:
        gray = color.rgb2gray(image_uint8)
    else:
        gray = image_uint8.astype(np.float64) / 255.0
    
    # Размытие
    blurred = filters.gaussian(gray, sigma=gaussian_sigma)
    
    # Бинаризация Otsu
    thresh = threshold_otsu(blurred)
    
    # Определение фона по углу изображения
    if blurred[0, 0] > thresh:
        binary = blurred < thresh
    else:
        binary = blurred > thresh
    
    # Морфология: закрытие разрывов
    selem = morphology.disk(closing_radius)
    closed_mask = morphology.binary_closing(binary, selem)
    
    # Удаление маленьких дырок внутри платы
    final_mask = morphology.remove_small_holes(closed_mask, area_threshold=hole_area_threshold)
    
    # Поиск контуров
    contours = measure.find_contours(final_mask, level=0.5)
    
    if not contours:
        print("[!] Контуры платы не найдены")
        return None
    
    # Самый длинный контур - внешняя граница
    main_contour = max(contours, key=lambda x: len(x))
    
    # Аппроксимация многоугольника
    poly_approx = measure.approximate_polygon(main_contour, tolerance=tolerance)
    
    # Убираем последнюю точку (она дублирует первую)
    all_corners = poly_approx[:-1] if len(poly_approx) > 1 else poly_approx
    
    # Находим 4 крайних угла
    if len(all_corners) >= 4:
        # Находим центр масс всех точек
        center = all_corners.mean(axis=0)
        
        # Вычисляем векторы от центра
        vectors = all_corners - center
        
        # Находим 4 крайние точки в направлениях: верх-левый, верх-правый, низ-правый, низ-левый
        # Используем комбинации координат для определения квадрантов
        
        # Верх-левый: минимальная y, минимальная x (относительно центра)
        top_left_idx = np.argmin(vectors[:, 0] + vectors[:, 1])  # Минимум суммы
        
        # Верх-правый: минимальная y, максимальная x
        top_right_idx = np.argmin(vectors[:, 0] - vectors[:, 1])  # Минимум y, максимум x
        
        # Низ-правый: максимальная y, максимальная x
        bottom_right_idx = np.argmax(vectors[:, 0] + vectors[:, 1])  # Максимум суммы
        
        # Низ-левый: максимальная y, минимальная x
        bottom_left_idx = np.argmax(vectors[:, 0] - vectors[:, 1])  # Максимум y, минимум x
        
        corners = np.array([
            all_corners[top_left_idx],
            all_corners[top_right_idx],
            all_corners[bottom_right_idx],
            all_corners[bottom_left_idx],
        ])
        print(f"[i] Найдено {len(all_corners)} точек контура, выбрано 4 крайних угла")
    elif len(all_corners) > 0:
        # Если меньше 4 точек, используем то что есть
        corners = all_corners
        print(f"[!] Найдено только {len(all_corners)} углов (нужно 4)")
    else:
        print("[!] Углы не найдены")
        return None
    
    if debug:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        axes[0].imshow(final_mask, cmap='gray')
        axes[0].set_title("Маска платы")
        axes[0].axis('off')
        
        # Показываем исходное изображение
        if image_uint8.ndim == 3:
            axes[1].imshow(image_uint8)
        else:
            axes[1].imshow(image_uint8, cmap='gray')
        axes[1].set_title(f"Углы платы (tolerance={tolerance})")
        axes[1].axis('off')
        
        # Рисуем контур
        if len(poly_approx) > 1:
            axes[1].plot(poly_approx[:, 1], poly_approx[:, 0], linewidth=2, color='#00FF00')
        
        # Рисуем углы
        if len(corners) > 0:
            axes[1].scatter(corners[:, 1], corners[:, 0], c='red', s=100, zorder=5)
            
            # Номера углов
            for i, (y, x) in enumerate(corners):
                axes[1].text(x + 5, y - 5, str(i), color='yellow', fontsize=12, weight='bold',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    return corners


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Вычисляет матрицу гомографии H (3x3) из соответствий точек.
    
    Parameters:
    -----------
    src_points : np.ndarray
        Исходные точки формы (N, 2) с координатами (y, x)
    dst_points : np.ndarray
        Целевые точки формы (N, 2) с координатами (y, x)
    
    Returns:
    --------
    np.ndarray
        Матрица гомографии 3x3
    """
    if len(src_points) != 4 or len(dst_points) != 4:
        raise ValueError("Нужно по 4 точки для вычисления гомографии")
    
    # Конвертируем (y, x) в (x, y) для гомографии
    src_xy = np.array([[p[1], p[0]] for p in src_points], dtype=np.float64)
    dst_xy = np.array([[p[1], p[0]] for p in dst_points], dtype=np.float64)
    
    A = []
    for (x, y), (u, v) in zip(src_xy, dst_xy):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A, dtype=np.float64)
    
    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)
    H = H / H[2, 2]
    return H


def warp_with_homography(image: Array, H: np.ndarray, output_shape: Tuple[int, int]) -> Array:
    """
    Преобразует изображение через гомографию H в output_shape.
    
    Parameters:
    -----------
    image : Array
        Исходное изображение
    H : np.ndarray
        Матрица гомографии 3x3
    output_shape : Tuple[int, int]
        Размер выходного изображения (height, width)
    
    Returns:
    --------
    Array
        Преобразованное изображение
    """
    height, width = output_shape
    is_grayscale = image.ndim == 2
    channels = image.shape[2] if image.ndim == 3 else 1
    
    inv_h = np.linalg.inv(H)
    yy, xx = np.indices((height, width))
    homog = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()], axis=0)
    mapped = inv_h @ homog
    mapped /= mapped[2:3]
    src_x = mapped[0].reshape(height, width)
    src_y = mapped[1].reshape(height, width)
    
    valid = (
        (src_x >= 0) & (src_x < image.shape[1] - 1) &
        (src_y >= 0) & (src_y < image.shape[0] - 1)
    )
    
    if is_grayscale:
        result = ndi.map_coordinates(image, [src_y, src_x], order=1, mode="nearest", cval=0.0)
        result[~valid] = 0.0
        return result
    else:
        result = np.zeros((height, width, channels), dtype=np.float32)
        for c in range(channels):
            result[..., c] = ndi.map_coordinates(
                image[..., c], [src_y, src_x], order=1, mode="nearest", cval=0.0
            )
        result[~valid] = 0.0
        return result


def _to_uint8_image(image: Array) -> np.ndarray:
    """Преобразует изображение к uint8 для работы с PIL."""
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.0:
        return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def subpixel(image: Image.Image, x: float, y: float) -> Tuple[int, int, int]:
    """
    Билинейная интерполяция для получения цвета пикселя в субпиксельных координатах.
    """
    width_img, height_img = image.size
    x_a, y_a = int(round(x)), int(round(y))
    x_b, y_b = x_a + 1, y_a
    x_c, y_c = x_a, y_a + 1
    x_d, y_d = x_a + 1, y_a + 1

    if (
        0 <= x_a < width_img and 0 <= y_a < height_img and
        0 <= x_b < width_img and 0 <= y_b < height_img and
        0 <= x_c < width_img and 0 <= y_c < height_img and
        0 <= x_d < width_img and 0 <= y_d < height_img
    ):
        p_a = image.getpixel((x_a, y_a))
        p_b = image.getpixel((x_b, y_b))
        p_c = image.getpixel((x_c, y_c))
        p_d = image.getpixel((x_d, y_d))

        p_e = tuple(a * (1 - (y - y_a)) + b * (1 - (y_c - y)) for a, b in zip(p_a, p_c))
        p_f = tuple(a * (1 - (y - y_b)) + b * (1 - (y_d - y)) for a, b in zip(p_b, p_d))
        return tuple(int(a * (1 - (x - x_a)) + b * (1 - (x_b - x))) for a, b in zip(p_e, p_f))

    x_safe = max(0, min(width_img - 1, int(round(x))))
    y_safe = max(0, min(height_img - 1, int(round(y))))
    return image.getpixel((x_safe, y_safe))


def apply_homography_img(image: Image.Image, H: np.ndarray, width: int, height: int) -> Image.Image:
    """
    Применяет матрицу гомографии к изображению с билинейной интерполяцией через PIL.
    """
    transformed_image = Image.new("RGB", (width, height))
    h_inv = np.linalg.inv(H)

    for y in range(height):
        for x in range(width):
            original_point = np.array([x, y, 1.0], dtype=np.float64)
            mapped = h_inv @ original_point
            mapped /= mapped[2]

            src_x, src_y = mapped[0], mapped[1]
            width_img, height_img = image.size
            if 0 <= src_x < width_img and 0 <= src_y < height_img:
                transformed_image.putpixel((x, y), subpixel(image, src_x, src_y))

    return transformed_image


def _rectified_size_from_ordered(corners: np.ndarray) -> Tuple[int, int]:
    """Оценивает размер растянутой платы по упорядоченным углам (y, x)."""
    if corners is None or len(corners) != 4:
        return 1, 1

    width = int(max(
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[3]),
    ))
    height = int(max(
        np.linalg.norm(corners[3] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
    ))
    return max(width, 1), max(height, 1)


def rectify_pcb_to_corners(
    image: Array,
    corners: np.ndarray | None,
    output_size: Tuple[int, int] | None = None,
) -> Tuple[Array, Array]:
    """
    Растягивает плату так, чтобы её углы совпадали с углами прямоугольного изображения.
    Возвращает растянутое изображение и маску платы.
    """
    if corners is None or len(corners) < 4:
        return image.copy(), np.ones(image.shape[:2], dtype=bool)

    ordered_corners = order_corners_clockwise(np.asarray(corners, dtype=np.float32))
    if ordered_corners.shape[0] != 4:
        return image.copy(), np.ones(image.shape[:2], dtype=bool)

    if output_size is None:
        width, height = _rectified_size_from_ordered(ordered_corners)
    else:
        width, height = output_size

    dst_corners = np.array([
        [0.0, 0.0],
        [0.0, width - 1],
        [height - 1, width - 1],
        [height - 1, 0.0],
    ], dtype=np.float32)

    H = compute_homography(ordered_corners, dst_corners)
    image_uint8 = _to_uint8_image(image)

    if image.ndim == 3:
        image_pil = Image.fromarray(image_uint8, mode="RGB")
    else:
        image_pil = Image.fromarray(image_uint8, mode="L").convert("RGB")

    rectified_pil = apply_homography_img(image_pil, H, width, height)
    rectified_array = np.asarray(rectified_pil).astype(np.float32) / 255.0

    if image.ndim == 2:
        rectified_array = np.dot(rectified_array[..., :3], [0.2989, 0.5870, 0.1140])

    if image.dtype == np.uint8:
        rectified_array = (rectified_array * 255.0).astype(np.uint8)
    else:
        rectified_array = rectified_array.astype(image.dtype)

    mask = np.ones((height, width), dtype=bool)
    return rectified_array, mask


def order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """
    Упорядочивает углы по часовой стрелке: верх-левый, верх-правый, низ-правый, низ-левый.
    
    Parameters:
    -----------
    corners : np.ndarray
        Массив углов формы (N, 2) с координатами (y, x)
    
    Returns:
    --------
    np.ndarray
        Упорядоченные углы (4, 2)
    """
    if len(corners) != 4:
        return corners
    
    # Находим центр
    center = corners.mean(axis=0)
    
    # Вычисляем углы относительно центра
    angles = np.arctan2(corners[:, 0] - center[0], corners[:, 1] - center[1])
    
    # Сортируем по углу
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]
    
    # Определяем верхние и нижние углы
    top = sorted_corners[sorted_corners[:, 0] < center[0]]
    bottom = sorted_corners[sorted_corners[:, 0] >= center[0]]
    
    if len(top) == 2 and len(bottom) == 2:
        # Сортируем верхние по x
        top = top[top[:, 1].argsort()]
        # Сортируем нижние по x
        bottom = bottom[bottom[:, 1].argsort()]
        # Порядок: верх-левый, верх-правый, низ-правый, низ-левый
        return np.array([top[0], top[1], bottom[1], bottom[0]])
    
    return sorted_corners


@dataclasses.dataclass
class DetectionConfig:
    blur_sigma: float = 1.5
    diff_threshold: float = 3.0  # в сигмах
    min_component_area: int = 120
    closing_size: int = 5
    max_keypoints: int = 2000
    # Параметры для нахождения границ платы
    corner_tolerance: float = 2.5
    corner_gaussian_sigma: float = 1.0
    corner_closing_radius: int = 3
    corner_hole_threshold: int = 5000
    rectify_to_corners: bool = True


def compute_difference_maps(
    reference: Array,
    test: Array,
    config: DetectionConfig,
    valid_mask: Array | None = None,
) -> Tuple[Array, Array, Array]:
    ref_gray = to_grayscale(reference)
    test_gray = to_grayscale(test)

    if valid_mask is None:
        valid_mask = np.ones_like(ref_gray, dtype=bool)

    test_gray = match_histogram(ref_gray, test_gray, mask=valid_mask)
    test_gray = normalize_intensity(ref_gray, test_gray, mask=valid_mask)

    ref_smooth = ndi.gaussian_filter(ref_gray, sigma=config.blur_sigma)
    test_smooth = ndi.gaussian_filter(test_gray, sigma=config.blur_sigma)

    diff = test_smooth - ref_smooth
    diff[~valid_mask] = 0.0
    sigma = np.std(diff[valid_mask]) + 1e-6
    threshold = config.diff_threshold * sigma

    extra_mask = (diff > threshold) & valid_mask
    missing_mask = (diff < -threshold) & valid_mask

    structure = np.ones((config.closing_size, config.closing_size))
    extra_mask = ndi.binary_closing(extra_mask, structure=structure)
    missing_mask = ndi.binary_closing(missing_mask, structure=structure)

    return diff, extra_mask, missing_mask


def extract_regions(mask: Array, diff: Array, min_area: int) -> List[dict]:
    labeled, num = ndi.label(mask)
    regions: List[dict] = []
    for region_id in range(1, num + 1):
        region_mask = labeled == region_id
        area = int(region_mask.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(region_mask)
        bbox = (xs.min(), ys.min(), xs.max(), ys.max())
        score = float(np.mean(np.abs(diff[region_mask])))
        center = (float(xs.mean()), float(ys.mean()))
        regions.append({
            "area": area,
            "bbox": bbox,
            "center": center,
            "score": score,
        })
    return regions


def visualize(reference: Array, test: Array, diff: Array, extra: List[dict], missing: List[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(reference)
    axes[0].set_title("Эталон")
    axes[0].axis("off")

    axes[1].imshow(test)
    axes[1].set_title("Тест")
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="coolwarm", vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    axes[2].set_title("Разница (тест - эталон)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for ax, regions, color, label in [
        (axes[1], extra, "lime", "Лишний"),
        (axes[0], missing, "red", "Отсутствует"),
    ]:
        for idx, region in enumerate(regions, 1):
            x0, y0, x1, y1 = region["bbox"]
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x0, y0 - 5, f"{label} #{idx}", color=color, fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.5, pad=2))

    plt.tight_layout()
    plt.show()


def detect_components(
    reference_path: str,
    test_path: str,
    config: DetectionConfig,
    show_corners: bool = True,
) -> None:
    reference = load_image(reference_path)
    test = load_image(test_path)
    reference_for_diff = reference
    
    print("\n=== Поиск границ плат ===")
    
    # Находим углы эталонной платы
    print("Поиск углов эталонной платы...")
    ref_corners = find_board_corners(
        reference,
        tolerance=config.corner_tolerance,
        gaussian_sigma=config.corner_gaussian_sigma,
        closing_radius=config.corner_closing_radius,
        hole_area_threshold=config.corner_hole_threshold,
        debug=show_corners,
    )
    
    # Находим углы тестовой платы
    print("Поиск углов тестовой платы...")
    test_corners = find_board_corners(
        test,
        tolerance=config.corner_tolerance,
        gaussian_sigma=config.corner_gaussian_sigma,
        closing_radius=config.corner_closing_radius,
        hole_area_threshold=config.corner_hole_threshold,
        debug=show_corners,
    )
    
    aligned_test = test.copy()
    valid_mask = np.ones(reference.shape[:2], dtype=bool)
    alignment_applied = False
    used_rectification = False
    
    # Выравниваем изображения, если найдено по 4 угла
    if ref_corners is not None and test_corners is not None:
        if len(ref_corners) == 4 and len(test_corners) == 4:
            try:
                print("\n=== Выравнивание изображений ===")
                
                # Упорядочиваем углы по часовой стрелке
                ref_ordered = order_corners_clockwise(ref_corners)
                test_ordered = order_corners_clockwise(test_corners)
                
                print(f"Эталонные углы (y, x):")
                for i, (y, x) in enumerate(ref_ordered):
                    print(f"  {i}: ({y:.1f}, {x:.1f})")
                
                print(f"Тестовые углы (y, x):")
                for i, (y, x) in enumerate(test_ordered):
                    print(f"  {i}: ({y:.1f}, {x:.1f})")
                
                if config.rectify_to_corners:
                    try:
                        ref_w, ref_h = _rectified_size_from_ordered(ref_ordered)
                        test_w, test_h = _rectified_size_from_ordered(test_ordered)
                        target_width = max(ref_w, test_w)
                        target_height = max(ref_h, test_h)

                        reference_rectified, ref_mask_rect = rectify_pcb_to_corners(
                            reference, ref_ordered, output_size=(target_width, target_height)
                        )
                        test_rectified, test_mask_rect = rectify_pcb_to_corners(
                            test, test_ordered, output_size=(target_width, target_height)
                        )

                        reference_for_diff = reference_rectified
                        aligned_test = test_rectified
                        valid_mask = ref_mask_rect & test_mask_rect
                        alignment_applied = True
                        used_rectification = True
                        print(f"[i] Платы растянуты до {target_width}x{target_height}")
                    except Exception as rect_exc:
                        print(f"[!] Ошибка при растяжении плат: {rect_exc}")

                if not alignment_applied:
                    # Вычисляем гомографию
                    H = compute_homography(test_ordered, ref_ordered)
                    print(f"Матрица гомографии вычислена")

                    # Применяем гомографию
                    aligned_test = warp_with_homography(test, H, reference.shape[:2])

                    # Создаём маску валидных пикселей
                    mask_ones = np.ones(test.shape[:2], dtype=np.float32)
                    mask_warp = warp_with_homography(mask_ones, H, reference.shape[:2])
                    valid_mask = mask_warp > 0.5

                    reference_for_diff = reference
                    alignment_applied = True
                    print(f"Выравнивание применено. Покрытие: {valid_mask.mean()*100:.1f}%")
                
                # Показываем результат выравнивания
                if show_corners:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(reference)
                    axes[0].set_title("Эталон")
                    axes[0].axis("off")
                    
                    axes[1].imshow(test)
                    axes[1].set_title("Тест (оригинал)")
                    axes[1].axis("off")
                    
                    axes[2].imshow(aligned_test)
                    axes[2].set_title("Тест (после выравнивания)")
                    axes[2].axis("off")
                    plt.suptitle("Результат выравнивания", fontsize=14)
                    plt.tight_layout()
                    plt.show()
                    
            except Exception as exc:
                print(f"[!] Ошибка при выравнивании: {exc}")
                import traceback
                traceback.print_exc()
                aligned_test = test
                valid_mask = np.ones(reference.shape[:2], dtype=bool)
        else:
            print(f"[!] Найдено не по 4 угла: эталон={len(ref_corners) if ref_corners is not None else 0}, тест={len(test_corners) if test_corners is not None else 0}")
            print("[!] Выравнивание пропущено, работаю с исходными изображениями")
    else:
        print("[!] Не удалось найти углы на одной из плат, работаю с исходными изображениями")
    
    print("\n=== Поиск отличий ===")
    if alignment_applied:
        if used_rectification:
            print("[i] Сравнение выполняется между растянутыми изображениями")
        else:
            print("[i] Сравнение выполняется между ВЫРОВНЕННЫМИ изображениями")
    else:
        print("[!] ВНИМАНИЕ: Сравнение выполняется между НЕВЫРОВНЕННЫМИ изображениями!")
    
    diff, extra_mask, missing_mask = compute_difference_maps(
        reference_for_diff,
        aligned_test,  # Используем выровненное изображение
        config,
        valid_mask=valid_mask,
    )
    extra_regions = extract_regions(extra_mask, diff, config.min_component_area)
    missing_regions = extract_regions(missing_mask, diff, config.min_component_area)

    print("\n=== Детектор компонентов ===")
    print(f"Лишние компоненты: {len(extra_regions)}")
    for idx, region in enumerate(extra_regions, 1):
        print(f"  #{idx}: центр={region['center']}, площадь={region['area']}, score={region['score']:.4f}")

    print(f"Отсутствующие компоненты: {len(missing_regions)}")
    for idx, region in enumerate(missing_regions, 1):
        print(f"  #{idx}: центр={region['center']}, площадь={region['area']}, score={region['score']:.4f}")

    visualize(reference_for_diff, aligned_test, diff, extra_regions, missing_regions)
    
def main() -> None:
    config = DetectionConfig(
        blur_sigma=2.0,
        diff_threshold=3.0,
        min_component_area=150,
        closing_size=5,
        corner_tolerance=2.5,
        corner_gaussian_sigma=1.0,
        corner_closing_radius=3,
        corner_hole_threshold=5000,
    )
    
    # Детекция компонентов с автоматическим поиском границ и выравниванием
    detect_components("qwerty.jpg", "qwerty.jpg", config, show_corners=True)


if __name__ == "__main__":
    main()
