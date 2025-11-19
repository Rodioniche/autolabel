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

    return H


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

    def preprocess_image(self, image):
        """
        Улучшенная предобработка изображения для детекции углов
        """
        if len(image.shape) == 3:
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image.copy()

        # Нормализация
        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0

        # 1. УВЕЛИЧЕНИЕ КОНТРАСТА - ключевое улучшение!
        # Адаптивное увеличение контраста
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # Сильное увеличение контраста
        gray_contrast = np.clip((gray - mean_val) * 3.0 + mean_val, 0, 1)

        # 2. ГИСТОГРАММНОЕ ВЫРАВНИВАНИЕ
        # Вычисляем кумулятивную гистограмму
        hist, bins = np.histogram(gray_contrast * 255, bins=256, range=(0, 255))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]

        # Применяем выравнивание гистограммы
        gray_equalized = np.interp(gray_contrast * 255, bins[:-1], cdf_normalized * 255) / 255.0

        # 3. ФИЛЬТРАЦИЯ ШУМА
        # Медианный фильтр для удаления шума
        gray_filtered = ndimage.median_filter(gray_equalized, size=5)

        # 4. УСИЛЕНИЕ ГРАНИЦ
        # Лапласиан для подчеркивания границ
        laplacian = ndimage.laplace(gray_filtered)
        gray_enhanced = np.clip(gray_filtered + 0.3 * laplacian, 0, 1)

        return gray_enhanced

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
        Улучшенный метод поиска углов платы
        """
        # ПРЕДОБРАБОТКА изображения
        processed_image = self.preprocess_image(image)

        height, width = processed_image.shape
        edges = []

        # Поиск границ по всем сторонам с увеличенной чувствительностью
        border_size = min(height, width) // 8  # Увеличили область поиска

        # Верхняя граница - более агрессивный поиск
        for j in range(border_size, width - border_size, 3):  # Уменьшили шаг
            for i in range(border_size, height // 3):
                # Используем обработанное изображение и более низкий порог
                if abs(int(processed_image[i, j] * 255) - int(processed_image[i - 1, j] * 255)) > 25:
                    edges.append((j, i))
                    break

        # Нижняя граница
        for j in range(border_size, width - border_size, 3):
            for i in range(height - border_size, height * 2 // 3, -1):
                if abs(int(processed_image[i, j] * 255) - int(processed_image[i - 1, j] * 255)) > 25:
                    edges.append((j, i))
                    break

        # Левая граница
        for i in range(border_size, height - border_size, 3):
            for j in range(border_size, width // 3):
                if abs(int(processed_image[i, j] * 255) - int(processed_image[i, j - 1] * 255)) > 25:
                    edges.append((j, i))
                    break

        # Правая граница
        for i in range(border_size, height - border_size, 3):
            for j in range(width - border_size, width * 2 // 3, -1):
                if abs(int(processed_image[i, j] * 255) - int(processed_image[i, j - 1] * 255)) > 25:
                    edges.append((j, i))
                    break

        # ДОПОЛНИТЕЛЬНО: поиск по угловым детекторам
        if len(edges) < 8:  # Если нашли мало точек, используем дополнительный метод
            additional_edges = self._find_corners_gradient(processed_image)
            edges.extend(additional_edges)

        # Кластеризация по углам
        if len(edges) >= 4:
            corners = self._cluster_corners(edges, width, height)
            ordered_corners = self._order_corners(corners)

            # Проверка качества углов
            if self._validate_corners(ordered_corners, width, height):
                return ordered_corners

        # Fallback: углы изображения с отступом
        print("Используем fallback углы")
        margin = min(width, height) // 15
        return [
            [margin, margin],
            [width - margin, margin],
            [width - margin, height - margin],
            [margin, height - margin]
        ]

    def _find_corners_gradient(self, image):
        """
        Дополнительный метод поиска углов через градиенты
        """
        height, width = image.shape
        corners = []

        # Вычисляем градиенты
        grad_x = ndimage.sobel(image, axis=1)
        grad_y = ndimage.sobel(image, axis=0)

        # Магнитуда градиента
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Ищем локальные максимумы градиента
        for i in range(10, height - 10, 5):
            for j in range(10, width - 10, 5):
                if gradient_magnitude[i, j] > 0.2:  # Порог для сильных градиентов
                    local_region = gradient_magnitude[i - 2:i + 3, j - 2:j + 3]
                    if gradient_magnitude[i, j] == np.max(local_region):
                        corners.append((j, i))

        return corners

    def _validate_corners(self, corners, width, height):
        """
        Проверяет, что углы образуют разумный прямоугольник
        """
        if len(corners) != 4:
            return False

        # Проверяем, что углы не слишком близко к краям
        margin = min(width, height) // 20
        for x, y in corners:
            if x < margin or x > width - margin or y < margin or y > height - margin:
                return False

        # Проверяем, что площадь не слишком мала
        area = self._calculate_polygon_area(corners)
        min_area = (width * height) * 0.3  # Минимум 30% от площади изображения
        if area < min_area:
            return False

        return True

    def _calculate_polygon_area(self, corners):
        """Вычисляет площадь многоугольника"""
        x = [p[0] for p in corners]
        y = [p[1] for p in corners]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

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

    def visualize_alignment(self, reference_image, test_image, H, aligned_corners, reference_corners, test_corners):
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

        # 3. Эталон с выровненными углами
        axes[0, 2].imshow(reference_image)
        if reference_corners:
            x_ref = [p[0] for p in reference_corners]
            y_ref = [p[1] for p in reference_corners]
            axes[0, 2].plot(x_ref, y_ref, 'go', markersize=8, label='Эталон')

        if aligned_corners and H is not None:
            x_aligned = [p[0] for p in aligned_corners]
            y_aligned = [p[1] for p in aligned_corners]
            axes[0, 2].plot(x_aligned, y_aligned, 'bo', markersize=8, label='Выровненные')
            for i in range(4):
                x1, y1 = aligned_corners[i]
                x2, y2 = aligned_corners[(i + 1) % 4]
                axes[0, 2].plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)

        axes[0, 2].set_title('Сравнение углов')
        axes[0, 2].legend()
        axes[0, 2].axis('off')

        # 4. Матрица гомографии
        if H is not None:
            axes[1, 0].text(0.1, 0.5, f'Матрица H:\n{np.array2string(H, precision=3)}',
                            fontsize=10, transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'Гомография не вычислена',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Матрица гомографии')
        axes[1, 0].axis('off')

        # 5. Ошибки выравнивания
        if H is not None and aligned_corners and reference_corners:
            errors = []
            for i in range(4):
                error = math.sqrt((aligned_corners[i][0] - reference_corners[i][0]) ** 2 +
                                  (aligned_corners[i][1] - reference_corners[i][1]) ** 2)
                errors.append(error)

            axes[1, 1].bar(range(4), errors, color=['red', 'blue', 'green', 'orange'])
            axes[1, 1].set_xlabel('Номер угла')
            axes[1, 1].set_ylabel('Ошибка (пиксели)')
            axes[1, 1].set_title('Ошибки выравнивания углов')
            axes[1, 1].set_xticks(range(4))
        else:
            axes[1, 1].text(0.5, 0.5, 'Нет данных для ошибок',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Ошибки выравнивания')
            axes[1, 1].axis('off')

        # 6. Общая информация
        info_text = f"Эталон: {len(reference_corners)} углов\nТест: {len(test_corners)} углов"
        if H is not None:
            info_text += f"\nГомография: УСПЕХ"
            if aligned_corners and reference_corners:
                avg_error = np.mean([math.sqrt((a[0] - r[0]) ** 2 + (a[1] - r[1]) ** 2)
                                     for a, r in zip(aligned_corners, reference_corners)])
                info_text += f"\nСредняя ошибка: {avg_error:.2f}px"
        else:
            info_text += f"\nГомография: ОШИБКА"

        axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Статистика')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    # Загрузка изображений
    reference_img = Image.open('фотка1.jpeg')
    test_img = Image.open('фотка2.jpeg')

    reference_array = np.array(reference_img)
    test_array = np.array(test_img)

    print(f"Эталон: {reference_array.shape}, Тест: {test_array.shape}")

    # Создаем детектор
    detector = PCBCornerDetector()

    # Выравниваем и сравниваем
    H, aligned_corners, ref_corners, test_corners = detector.align_and_compare(
        reference_array, test_array
    )

    # Визуализируем результаты
    detector.visualize_alignment(
        reference_array, test_array, H, aligned_corners, ref_corners, test_corners
    )


if __name__ == "__main__":
    main()