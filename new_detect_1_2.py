import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from matplotlib.path import Path
import pickle
from pathlib import Path as FilePath

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[Предупреждение] Модуль cv2 (OpenCV) не установлен. Устранение дисторсии недоступно.")

try:
    from skimage import color, filters, measure, morphology
    from skimage.filters import threshold_otsu
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[Предупреждение] Модуль scikit-image не установлен. Продвинутый поиск углов недоступен.")


# ========== Функции для устранения дисторсии ==========

def undistort_image(image: np.ndarray, calibration_file: str = 'camera_calibration.pkl') -> np.ndarray:
    """
    Устраняет дисторсию изображения с использованием калибровочных данных камеры.
    
    Parameters:
    image: numpy array - исходное изображение (BGR или RGB)
    calibration_file: str - путь к файлу калибровки
    
    Returns:
    numpy array - изображение без дисторсии
    """
    if not CV2_AVAILABLE:
        print("   [Пропущено] OpenCV недоступен")
        return image
    
    calib_path = FilePath(calibration_file)
    if not calib_path.exists():
        print(f"   [Пропущено] Файл калибровки не найден")
        return image
    
    try:
        # Загрузка калибровочных данных
        with open(calibration_file, 'rb') as f:
            mtx, dist = pickle.load(f)
        
        h, w = image.shape[:2]
        
        # Получаем оптимальную матрицу камеры
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # Устранение дисторсии
        dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
        
        # Обрезаем изображение по ROI
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
        
        print(f"   ✓ Дисторсия устранена: {image.shape} → {dst.shape}")
        return dst
    
    except Exception as e:
        print(f"   [Ошибка] {e}")
        return image


# ========== Эквализация гистограммы в LAB пространстве ==========

def rgb_to_lab_cv(rgb_uint8):
    """Конвертирует RGB в LAB через OpenCV"""
    if not CV2_AVAILABLE:
        return rgb_uint8
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab


def lab_to_rgb_cv(lab_uint8):
    """Конвертирует LAB в RGB через OpenCV"""
    if not CV2_AVAILABLE:
        return lab_uint8
    bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def equalize_L_numpy(L_uint8):
    """
    Глобальная эквализация L-канала (0..255) через CDF.
    Возвращает L_eq, hist(L_eq), cdf(L_eq).
    """
    # CDF для построения LUT
    hist, _ = np.histogram(L_uint8.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    
    # Нормализация CDF в 0..255
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    lut = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # Применяем LUT
    L_eq = lut[L_uint8]
    
    # CDF для отрисовки считаем заново по L_eq
    hist_after, _ = np.histogram(L_eq.flatten(), 256, [0, 256])
    cdf_after = hist_after.cumsum()
    
    return L_eq, hist_after, cdf_after


def apply_clahe_L_lab(lab_uint8, clipLimit=3.0, tileGridSize=(8, 8)):
    """
    CLAHE только по L-каналу в LAB.
    Возвращает lab_clahe и (hist_L_after, cdf_L_after).
    """
    if not CV2_AVAILABLE:
        return lab_uint8, (None, None)
    
    L = lab_uint8[:, :, 0]
    A = lab_uint8[:, :, 1]
    B = lab_uint8[:, :, 2]
    
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    L_clahe = clahe.apply(L)
    
    lab_clahe = np.stack([L_clahe, A, B], axis=2).astype(np.uint8)
    
    # CDF для отрисовки – по результату CLAHE
    hist_L, _ = np.histogram(L_clahe.flatten(), 256, [0, 256])
    cdf_L = hist_L.cumsum()
    
    return lab_clahe, (hist_L, cdf_L)


def equalize_histogram_lab(image, method='clahe', clipLimit=3.0, tileGridSize=(8, 8)):
    """
    Эквализирует гистограмму изображения в LAB пространстве.
    
    Parameters:
    -----------
    image : numpy array
        Исходное изображение (RGB, float32 [0,1] или uint8 [0,255])
    method : str
        'clahe' - адаптивная эквализация (рекомендуется)
        'global' - глобальная эквализация
        'none' - без эквализации
    clipLimit : float
        Порог ограничения контраста для CLAHE
    tileGridSize : tuple
        Размер сетки для CLAHE
    
    Returns:
    --------
    numpy array
        Изображение с эквализированной гистограммой (тот же формат что и входное)
    """
    if not CV2_AVAILABLE:
        print("   [Пропущено] OpenCV недоступен")
        return image
    
    if method == 'none':
        return image
    
    # Сохраняем исходный формат
    was_float = image.dtype in [np.float32, np.float64]
    original_dtype = image.dtype
    
    # Конвертируем в uint8
    if was_float:
        if image.max() <= 1.0:
            img_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)
    
    # Обрабатываем RGBA -> RGB
    if img_uint8.ndim == 3 and img_uint8.shape[2] == 4:
        img_uint8 = img_uint8[:, :, :3]
    
    # Grayscale -> RGB для LAB конверсии
    if img_uint8.ndim == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        was_grayscale = True
    else:
        was_grayscale = False
    
    try:
        # RGB -> LAB
        lab = rgb_to_lab_cv(img_uint8)
        
        if method == 'global':
            # Глобальная эквализация L-канала
            L_eq, _, _ = equalize_L_numpy(lab[:, :, 0])
            lab_eq = lab.copy()
            lab_eq[:, :, 0] = L_eq
        elif method == 'clahe':
            # CLAHE по L-каналу
            lab_eq, _ = apply_clahe_L_lab(lab, clipLimit=clipLimit, tileGridSize=tileGridSize)
        else:
            lab_eq = lab
        
        # LAB -> RGB
        rgb_eq = lab_to_rgb_cv(lab_eq)
        
        # Конвертируем обратно в grayscale если нужно
        if was_grayscale:
            rgb_eq = cv2.cvtColor(rgb_eq, cv2.COLOR_RGB2GRAY)
        
        # Конвертируем обратно в исходный формат
        if was_float:
            result = rgb_eq.astype(np.float32) / 255.0
            if original_dtype == np.float64:
                result = result.astype(np.float64)
        else:
            result = rgb_eq.astype(original_dtype)
        
        method_name = {
            'global': 'глобальная эквализация',
            'clahe': f'CLAHE (clip={clipLimit}, grid={tileGridSize})',
            'none': 'без эквализации'
        }
        print(f"   ✓ Эквализация гистограммы: {method_name.get(method, method)}")
        
        return result
        
    except Exception as e:
        print(f"   [Ошибка] {e}")
        return image


# ========== Продвинутый поиск углов платы (из corner_detection.py) ==========

def find_board_corners(
    image,
    tolerance=2.5,
    gaussian_sigma=1.0,
    closing_radius=3,
    hole_area_threshold=5000,
    debug=False,
):
    """
    Находит углы платы на изображении используя scikit-image.
    Продвинутый алгоритм с морфологией и бинаризацией Otsu.
    
    Parameters:
    -----------
    image : numpy array
        Изображение в формате [0, 1] или uint8, RGB или grayscale
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
    list or None
        Список углов [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] или None если не найдено
    """
    if not SKIMAGE_AVAILABLE:
        print("   [Пропущено] scikit-image недоступен, используется fallback алгоритм")
        return None
    
    # Конвертируем в uint8 для skimage (если нужно)
    if image.max() <= 1.0:
        image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    
    # Обработка RGBA изображений (конвертируем в RGB)
    if image_uint8.ndim == 3 and image_uint8.shape[2] == 4:
        # RGBA -> RGB (игнорируем альфа-канал)
        image_uint8 = image_uint8[:, :, :3]
    
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
        print("   [!] Контуры платы не найдены")
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
        # В skimage контуры возвращаются в формате (y, x), нужно конвертировать в (x, y)
        
        # Верх-левый: минимальная y, минимальная x (относительно центра)
        top_left_idx = np.argmin(vectors[:, 0] + vectors[:, 1])
        
        # Верх-правый: минимальная y, максимальная x
        top_right_idx = np.argmin(vectors[:, 0] - vectors[:, 1])
        
        # Низ-правый: максимальная y, максимальная x
        bottom_right_idx = np.argmax(vectors[:, 0] + vectors[:, 1])
        
        # Низ-левый: максимальная y, минимальная x
        bottom_left_idx = np.argmax(vectors[:, 0] - vectors[:, 1])
        
        # skimage возвращает (y, x), конвертируем в (x, y)
        corners_yx = np.array([
            all_corners[top_left_idx],
            all_corners[top_right_idx],
            all_corners[bottom_right_idx],
            all_corners[bottom_left_idx],
        ])
        
        # Конвертируем из (y, x) в (x, y)
        corners = [(float(x), float(y)) for y, x in corners_yx]
        
        print(f"   ✓ Найдено {len(all_corners)} точек контура, выбрано 4 крайних угла")
    elif len(all_corners) > 0:
        # Если меньше 4 точек, используем то что есть
        corners = [(float(x), float(y)) for y, x in all_corners]
        print(f"   [!] Найдено только {len(all_corners)} углов (нужно 4)")
    else:
        print("   [!] Углы не найдены")
        return None
    
    if debug:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        axes[0].imshow(final_mask, cmap='gray')
        axes[0].set_title("Маска платы (Otsu + морфология)")
        axes[0].axis('off')
        
        # Показываем исходное изображение
        if image_uint8.ndim == 3:
            axes[1].imshow(image_uint8)
        else:
            axes[1].imshow(image_uint8, cmap='gray')
        axes[1].set_title(f"Углы платы (tolerance={tolerance})")
        axes[1].axis('off')
        
        # Рисуем контур (конвертируем обратно в (x, y))
        if len(poly_approx) > 1:
            axes[1].plot(poly_approx[:, 1], poly_approx[:, 0], linewidth=2, color='#00FF00')
        
        # Рисуем углы
        if len(corners) > 0:
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            axes[1].scatter(xs, ys, c='red', s=100, zorder=5)
            
            # Номера углов
            for i, (x, y) in enumerate(corners):
                axes[1].text(x + 5, y - 5, str(i), color='yellow', fontsize=12, weight='bold',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    return corners if len(corners) == 4 else None


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
                        grid_size=8, diff_threshold=0.05):
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


# ========== SSIM сравнение (из diff_ssim.py) ==========

def compute_ssim(img1, img2):
    """Вычисляет SSIM между двумя изображениями"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    covariance = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * covariance + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

    return numerator / denominator if denominator != 0 else 1.0


def compare_images_with_ssim(ref_img, test_img, block_size=16, diff_thresh=30, ssim_thresh=0.75):
    """
    Сравнивает изображения используя SSIM по блокам.
    
    Parameters:
    -----------
    ref_img : PIL Image
        Эталонное изображение
    test_img : PIL Image
        Тестовое изображение
    block_size : int
        Размер блока для сравнения
    diff_thresh : int
        Порог различий пикселей
    ssim_thresh : float
        Порог SSIM (ниже - дефект)
    
    Returns:
    --------
    tuple : (result_img, defects_img)
        result_img - изображение с обводкой дефектов
        defects_img - только дефекты на белом фоне
    """
    from PIL import ImageDraw
    
    # Конвертируем в grayscale для анализа
    ref_gray = ref_img.convert("L")
    test_gray = test_img.convert("L")
    test_color = test_img.convert("RGB")
    
    ref_array = np.array(ref_gray)
    test_array = np.array(test_gray)
    
    result_img = test_color.copy()
    draw = ImageDraw.Draw(result_img)
    
    defects_img = Image.new("RGB", result_img.size, color=(255, 255, 255))
    defects_pixels = defects_img.load()
    test_pixels_color = test_color.load()
    
    height, width = ref_array.shape
    defect_blocks = []
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block_ref = ref_array[y:y+block_size, x:x+block_size]
            block_test = test_array[y:y+block_size, x:x+block_size]
            
            if block_ref.shape != (block_size, block_size) or block_test.shape != (block_size, block_size):
                continue
            
            diff_pixels = np.abs(block_ref - block_test)
            num_diffs = np.sum(diff_pixels > diff_thresh)
            
            if num_diffs > (block_size * block_size) // 4:
                ssim_score = compute_ssim(block_ref, block_test)
                if ssim_score < ssim_thresh:
                    # Обводим на оригинале
                    draw.rectangle([x, y, x+block_size-1, y+block_size-1], outline=(255, 0, 0), width=2)
                    
                    # Копируем цветной блок на белый фон
                    for j in range(block_size):
                        for i in range(block_size):
                            if (y + j < height) and (x + i < width):
                                defects_pixels[x + i, y + j] = test_pixels_color[x + i, y + j]
                    
                    defect_blocks.append({
                        'x': x, 'y': y,
                        'ssim': ssim_score,
                        'diff_pixels': int(num_diffs)
                    })
    
    return result_img, defects_img, defect_blocks


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

    def find_pcb_corners_simple(self, image, debug=False):
        """
        Метод поиска углов платы используя продвинутый алгоритм из corner_detection.py.
        
        Использует:
        - Бинаризацию Otsu (адаптивный порог)
        - Морфологические операции (закрытие разрывов)
        - Удаление дырок внутри платы
        - Поиск контуров через scikit-image
        - Аппроксимацию многоугольника
        
        Если scikit-image недоступен, использует fallback алгоритм.
        """
        # Пытаемся использовать продвинутый алгоритм
        corners = find_board_corners(
            image,
            tolerance=2.5,
            gaussian_sigma=1.0,
            closing_radius=3,
            hole_area_threshold=5000,
            debug=debug
        )
        
        if corners and len(corners) == 4:
            return corners
        
        # Fallback: если продвинутый алгоритм не сработал
        print("   [Fallback] Используется простой алгоритм поиска углов")
        
        # Конвертируем в grayscale
        if len(image.shape) == 3:
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image.copy()
        
        # Нормализуем в диапазон 0-255
        if gray.max() <= 1.0:
            img_array = (gray * 255).astype(np.uint8)
        else:
            img_array = gray.astype(np.uint8)
        
        # Бинаризация
        threshold = 200
        binary = img_array < threshold
        
        # Находим граничные точки
        boundaries = self._find_boundaries(binary)
        
        if not boundaries:
            print("   [Предупреждение] Не удалось найти границы платы! Используем углы изображения.")
            h, w = image.shape[:2]
            margin = 50
            return [
                [margin, margin],
                [w - margin, margin],
                [w - margin, h - margin],
                [margin, h - margin]
            ]
        
        # Выпуклая оболочка
        hull_points = self._convex_hull(boundaries)
        
        # Упрощение до 4 точек
        corners = self._simplify_to_quadrangle(hull_points, binary.shape)
        
        if corners and len(corners) == 4:
            print(f"   ✓ Найдено {len(boundaries)} граничных точек, выпуклая оболочка: {len(hull_points)} точек")
            # Упорядочиваем углы
            corners = self._order_corners(corners)
            return corners
        
        # Fallback
        h, w = image.shape[:2]
        margin = 50
        return [
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ]
    
    def _find_boundaries(self, binary):
        """Находит граничные точки на бинарном изображении"""
        boundaries = []
        h, w = binary.shape
        for y in range(1, h-1):
            for x in range(1, w-1):
                if binary[y, x]:
                    neighbors = [
                        binary[y-1, x], binary[y+1, x],
                        binary[y, x-1], binary[y, x+1]
                    ]
                    if not all(neighbors):
                        boundaries.append((x, y))
        return boundaries
    
    def _convex_hull(self, points):
        """Вычисляет выпуклую оболочку методом Грэхема"""
        points = sorted(set(points))
        if len(points) <= 1:
            return points
        
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
        # Нижняя часть оболочки
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        # Верхняя часть оболочки
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        return lower[:-1] + upper[:-1]
    
    def _simplify_to_quadrangle(self, hull, img_shape, min_distance_ratio=0.05):
        """Упрощает выпуклую оболочку до 4 угловых точек"""
        n = len(hull)
        if n <= 4:
            return hull
        
        img_diagonal = np.sqrt(img_shape[0]**2 + img_shape[1]**2)
        min_distance = img_diagonal * min_distance_ratio
        
        best_points = None
        max_area = 0
        
        # Оптимизация для большого количества точек
        if n > 20:
            step = max(1, n // 16)
            hull = hull[::step] + hull[-4:]
            # ОБНОВЛЯЕМ n после изменения hull!
            n = len(hull)
        
        def polygon_area(pts):
            """Вычисляет площадь многоугольника"""
            pts_loop = pts + [pts[0]]
            area = 0
            for i in range(len(pts)):
                x1, y1 = pts_loop[i]
                x2, y2 = pts_loop[i+1]
                area += (x1 * y2 - x2 * y1)
            return 0.5 * abs(area)
        
        # Перебираем все комбинации 4 точек
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        pts = [hull[i], hull[j], hull[k], hull[l]]
                        
                        # Проверяем минимальное расстояние между точками
                        distances = []
                        for idx1 in range(4):
                            for idx2 in range(idx1+1, 4):
                                dist = np.sqrt((pts[idx1][0]-pts[idx2][0])**2 +
                                              (pts[idx1][1]-pts[idx2][1])**2)
                                distances.append(dist)
                        if min(distances) < min_distance:
                            continue
                        
                        area = polygon_area(pts)
                        if area > max_area:
                            max_area = area
                            best_points = pts
        
        return best_points if best_points else hull[:4]

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
    print("="*70)
    print("СРАВНЕНИЕ ПЕЧАТНЫХ ПЛАТ")
    print("="*70)
    
    # Загрузка изображений
    print("\n📂 ЗАГРУЗКА ИЗОБРАЖЕНИЙ")
    print("-"*70)
    reference_img = Image.open('etalon_1.jpg')
    test_img = Image.open('test_1.jpg')

    reference_array = np.array(reference_img)
    test_array = np.array(test_img)

    print(f"Эталон: {reference_array.shape}")
    print(f"Тест:   {test_array.shape}")

    # ========== ЭТАП 1: УСТРАНЕНИЕ ДИСТОРСИИ ==========
    print("\n🔧 ЭТАП 1: УСТРАНЕНИЕ ДИСТОРСИИ")
    print("-"*70)
    print("Эталонное изображение:")
    reference_array = undistort_image(reference_array)
    print("Тестовое изображение:")
    test_array = undistort_image(test_array)
    
    print(f"\nРезультат после этапа 1:")
    print(f"  Эталон: {reference_array.shape}")
    print(f"  Тест:   {test_array.shape}")

    # ========== ЭТАП 2: ПОИСК УГЛОВ (с эквализацией для улучшения) ==========
    print("\n🔍 ЭТАП 2: ПОИСК УГЛОВ ПЛАТ")
    print("-"*70)
    
    # Создаем эквализированные копии только для поиска углов
    print("Применяем эквализацию для улучшения поиска углов...")
    reference_eq = equalize_histogram_lab(reference_array.copy(), method='clahe', clipLimit=3.0, tileGridSize=(8, 8))
    test_eq = equalize_histogram_lab(test_array.copy(), method='clahe', clipLimit=3.0, tileGridSize=(8, 8))
    
    detector = PCBCornerDetector()

    # Находим углы на эквализированных изображениях
    print("\nПоиск углов на эквализированных изображениях:")
    H, aligned_corners, ref_corners, test_corners = detector.align_and_compare(
        reference_eq, test_eq
    )
    
    print("\n[Важно] Углы найдены на эквализированных изображениях.")
    print("        Дальнейшая обработка использует ОРИГИНАЛЬНЫЕ изображения без эквализации.")

    if H is not None:
        # ========== ЭТАП 4: РАСТЯЖЕНИЕ ПЛАТ ==========
        print("\n📐 ЭТАП 4: РАСТЯЖЕНИЕ ПЛАТ ДО УГЛОВ ИЗОБРАЖЕНИЯ")
        print("-"*70)
        
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
        
        print(f"Эталон после растяжения: {ref_rectified.shape}")
        print(f"Тест после растяжения:   {test_rectified.shape}")
        print(f"Целевой размер: ({target_height}, {target_width})")
        
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
        
        print(f"✓ Финальный размер обеих плат: {reference_pcb_region.shape}")
        print("✓ Платы растянуты до углов изображения и готовы к сравнению")
        
        # ========== СОХРАНЕНИЕ МАЛЕНЬКИХ ИЗОБРАЖЕНИЙ ==========
        print("\n💾 СОХРАНЕНИЕ РАСТЯНУТЫХ ПЛАТ")
        print("-"*70)
        
        # Создаем папку small_photos
        small_photos_dir = FilePath("small_photos")
        small_photos_dir.mkdir(exist_ok=True)
        
        # Конвертируем в uint8 для сохранения
        if reference_pcb_region.dtype != np.uint8:
            if reference_pcb_region.max() <= 1.0:
                ref_save = (np.clip(reference_pcb_region, 0.0, 1.0) * 255.0).astype(np.uint8)
                test_save = (np.clip(transformed_test_pcb, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                ref_save = np.clip(reference_pcb_region, 0, 255).astype(np.uint8)
                test_save = np.clip(transformed_test_pcb, 0, 255).astype(np.uint8)
        else:
            ref_save = reference_pcb_region
            test_save = transformed_test_pcb
        
        # Сохраняем изображения
        ref_path = small_photos_dir / "reference_rectified.jpg"
        test_path = small_photos_dir / "test_rectified.jpg"
        
        Image.fromarray(ref_save).save(ref_path, quality=95)
        Image.fromarray(test_save).save(test_path, quality=95)
        
        print(f"✓ Эталон сохранен: {ref_path}")
        print(f"✓ Тест сохранен: {test_path}")
        
        # ========== ЭТАП 5: СРАВНЕНИЕ ПЛАТ ==========
        print("\n📊 ЭТАП 5: СРАВНЕНИЕ ПЛАТ")
        print("-"*70)

        # Визуализируем результаты
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Исходные изображения с углами (БЕЗ эквализации, углы найдены на эквализированных)
        axes[0, 0].imshow(reference_array)
        x_ref = [p[0] for p in ref_corners]
        y_ref = [p[1] for p in ref_corners]
        axes[0, 0].plot(x_ref, y_ref, 'ro-', markersize=8)
        axes[0, 0].set_title('Эталон\n(дисторсия убрана, БЕЗ эквализации)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(test_array)
        x_test = [p[0] for p in test_corners]
        y_test = [p[1] for p in test_corners]
        axes[0, 1].plot(x_test, y_test, 'ro-', markersize=8)
        axes[0, 1].set_title('Тест\n(дисторсия убрана, БЕЗ эквализации)')
        axes[0, 1].axis('off')

        # 2. Растянутые платы (углы платы = углы изображения)
        axes[1, 0].imshow(reference_pcb_region)
        # Показываем углы изображения (которые теперь совпадают с углами платы)
        h, w = reference_pcb_region.shape[:2]
        corners_img = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]
        x_corners = [p[0] for p in corners_img]
        y_corners = [p[1] for p in corners_img]
        axes[1, 0].plot(x_corners + [x_corners[0]], y_corners + [y_corners[0]], 'go-', markersize=6, linewidth=2)
        axes[1, 0].set_title(f'Эталон РАСТЯНУТ\n(этап 4) {reference_pcb_region.shape}')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(transformed_test_pcb)
        # Показываем углы изображения
        h_t, w_t = transformed_test_pcb.shape[:2]
        corners_img_t = [(0, 0), (w_t-1, 0), (w_t-1, h_t-1), (0, h_t-1)]
        x_corners_t = [p[0] for p in corners_img_t]
        y_corners_t = [p[1] for p in corners_img_t]
        axes[1, 1].plot(x_corners_t + [x_corners_t[0]], y_corners_t + [y_corners_t[0]], 'bo-', markersize=6, linewidth=2)
        axes[1, 1].set_title(f'Тест РАСТЯНУТ\n(этап 4) {transformed_test_pcb.shape}')
        axes[1, 1].axis('off')

        # 3. Карта различий
        axes[1, 2].imshow(transformed_test_pcb)
        axes[1, 2].set_title(f'Готово к сравнению\n{transformed_test_pcb.shape}')
        axes[1, 2].axis('off')

        # 4. Эталонная для сравнения
        axes[0, 2].imshow(reference_pcb_region)
        axes[0, 2].set_title('Эталон для сравнения')
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
            grid_size=14,
            diff_threshold=0.07
        )
        
        # ========== SSIM СРАВНЕНИЕ ==========
        print("\n🔬 SSIM АНАЛИЗ ДЕФЕКТОВ")
        print("-"*70)
        
        try:
            # Загружаем сохраненные изображения
            ref_pil = Image.open(ref_path)
            test_pil = Image.open(test_path)
            
            # Применяем SSIM анализ
            result_with_boxes, defects_only, defect_blocks = compare_images_with_ssim(
                ref_pil, test_pil,
                block_size=16,
                diff_thresh=30,
                ssim_thresh=0.75
            )
            
            # Сохраняем результаты
            result_path = small_photos_dir / "ssim_result_with_boxes.jpg"
            defects_path = small_photos_dir / "ssim_defects_only.jpg"
            
            result_with_boxes.save(result_path, quality=95)
            defects_only.save(defects_path, quality=95)
            
            print(f"✓ Найдено дефектных блоков (SSIM): {len(defect_blocks)}")
            print(f"✓ Результат с обводкой: {result_path}")
            print(f"✓ Только дефекты: {defects_path}")
            
            if defect_blocks:
                print("\nДетали дефектов:")
                for i, block in enumerate(defect_blocks[:5], 1):  # Показываем первые 5
                    print(f"  #{i}: позиция ({block['x']}, {block['y']}), SSIM={block['ssim']:.3f}, diff_pixels={block['diff_pixels']}")
                if len(defect_blocks) > 5:
                    print(f"  ... и еще {len(defect_blocks) - 5} блоков")
        
        except Exception as e:
            print(f"[!] Ошибка при SSIM анализе: {e}")

        # ========== ЭТАП 6: ВИЗУАЛИЗАЦИЯ ==========
        print("\n📈 ЭТАП 6: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
        print("-"*70)
        print("Открытие графиков...")
        
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
        
        print("\n" + "="*70)
        print("✅ СРАВНЕНИЕ ЗАВЕРШЕНО")
        print("="*70)

    else:
        print("\n❌ Не удалось вычислить гомографию")
        print("="*70)


if __name__ == "__main__":
    main()