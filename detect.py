import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from matplotlib.path import Path


class PCBComparator:
    def __init__(self):
        self.reference_corners = None
        self.reference_image = None

    def preprocess_image(self, image):
        """Предобработка изображения"""
        if len(image.shape) == 3:
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image.copy()

        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0

        # Увеличение контраста
        gray = np.clip((gray - gray.mean()) * 2.0 + gray.mean(), 0, 1)

        # Медианный фильтр
        gray = ndimage.median_filter(gray, size=3)

        return gray

    def find_pcb_corners(self, image):
        """
        Находит углы платы через анализ градиентов
        """
        if len(image.shape) == 3:
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image.copy()

        height, width = gray.shape
        edges = []

        # Поиск границ по всем сторонам
        border_size = min(height, width) // 10  # 10% от размера

        # Верхняя граница
        for j in range(border_size, width - border_size, 5):
            for i in range(border_size, height // 3):
                if abs(int(gray[i, j]) - int(gray[i - 1, j])) > 30:
                    edges.append((j, i))  # (x, y)
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
            [margin, margin],  # верхний левый
            [width - margin, margin],  # верхний правый
            [width - margin, height - margin],  # нижний правый
            [margin, height - margin]  # нижний левый
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

        # Сортируем по Y
        corners_sorted = sorted(corners, key=lambda p: p[1])
        top = sorted(corners_sorted[:2], key=lambda p: p[0])
        bottom = sorted(corners_sorted[2:], key=lambda p: p[0])

        return [top[0], top[1], bottom[1], bottom[0]]

    def create_pcb_mask(self, image_shape, corners):
        """Создает маску платы"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        if len(corners) == 4:
            poly_path = Path(corners)
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            points = np.vstack((x.flatten(), y.flatten())).T
            mask = poly_path.contains_points(points).reshape(h, w)

        return mask

    def set_reference(self, image_path):
        """Устанавливает эталонное изображение"""
        img = Image.open(image_path)
        self.reference_image = np.array(img)
        self.reference_corners = self.find_pcb_corners(self.reference_image)
        print("Эталонное изображение загружено")
        return self.reference_corners

    def compare_with_reference(self, test_image_path, grid_size=8, diff_threshold=0.6):
        """Сравнивает тестовое изображение с эталоном"""
        if self.reference_image is None:
            raise ValueError("Сначала установите эталонное изображение")

        # Загружаем тестовое изображение
        test_img = Image.open(test_image_path)
        test_image = np.array(test_img)

        # Находим углы тестовой платы
        test_corners = self.find_pcb_corners(test_image)

        # Создаем маски
        ref_mask = self.create_pcb_mask(self.reference_image.shape, self.reference_corners)
        test_mask = self.create_pcb_mask(test_image.shape, test_corners)

        # Преобразуем в grayscale для сравнения
        if len(self.reference_image.shape) == 3:
            ref_gray = np.dot(self.reference_image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            ref_gray = self.reference_image.copy()

        if len(test_image.shape) == 3:
            test_gray = np.dot(test_image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            test_gray = test_image.copy()

        # Нормализуем
        if ref_gray.max() > 1.0:
            ref_gray = ref_gray.astype(np.float32) / 255.0
        if test_gray.max() > 1.0:
            test_gray = test_gray.astype(np.float32) / 255.0

        # Сравниваем по сетке
        comparison_results = self._compare_by_grid(
            ref_gray, test_gray, ref_mask, test_mask, grid_size, diff_threshold
        )

        # Визуализируем результаты
        self._visualize_comparison(self.reference_image, test_image,
                                   self.reference_corners, test_corners,
                                   comparison_results, grid_size, diff_threshold)

        return comparison_results

    def _compare_by_grid(self, ref_gray, test_gray, ref_mask, test_mask, grid_size, diff_threshold):
        """Сравнивает изображения по сетке"""
        height, width = ref_gray.shape

        # Определяем границы платы
        ref_y, ref_x = np.where(ref_mask)
        if len(ref_y) == 0:
            return []

        min_y, max_y = np.min(ref_y), np.max(ref_y)
        min_x, max_x = np.min(ref_x), np.max(ref_x)

        pcb_height = max_y - min_y
        pcb_width = max_x - min_x

        # Размер сегмента
        segment_h = pcb_height // grid_size
        segment_w = pcb_width // grid_size

        results = []

        for i in range(grid_size):
            for j in range(grid_size):
                # Координаты сегмента
                y_start = min_y + i * segment_h
                y_end = min_y + (i + 1) * segment_h
                x_start = min_x + j * segment_w
                x_end = min_x + (j + 1) * segment_w

                # Извлекаем сегменты
                ref_segment = ref_gray[y_start:y_end, x_start:x_end]
                test_segment = test_gray[y_start:y_end, x_start:x_end]

                # Маски сегментов
                ref_seg_mask = ref_mask[y_start:y_end, x_start:x_end]
                test_seg_mask = test_mask[y_start:y_end, x_start:x_end]

                # Проверяем что сегменты не пустые
                if (ref_segment.size > 0 and test_segment.size > 0 and
                        ref_segment.shape == test_segment.shape):
                    # Вычисляем разницу
                    difference = self._calculate_difference(ref_segment, test_segment,
                                                            ref_seg_mask, test_seg_mask)

                    results.append({
                        'grid_pos': (i, j),
                        'coords': (x_start, y_start, x_end, y_end),
                        'difference': difference,
                        'is_defect': difference > diff_threshold  # порог можно настроить
                    })

        return results

    def _calculate_difference(self, ref_segment, test_segment, ref_mask, test_mask):
        """Усиленное сравнение с акцентом на максимальные различия"""
        valid_mask = ref_mask & test_mask
        if np.sum(valid_mask) == 0:
            return 0

        # 1. ОСНОВНАЯ ГИСТОГРАММА (30%)
        ref_masked = ref_segment[valid_mask]
        test_masked = test_segment[valid_mask]

        hist1 = np.histogram(ref_masked, bins=16, range=(0, 1))[0]
        hist2 = np.histogram(test_masked, bins=16, range=(0, 1))[0]

        hist1 = hist1 / (np.sum(hist1) + 1e-8)
        hist2 = hist2 / (np.sum(hist2) + 1e-8)

        similarity = np.sum(np.sqrt(hist1 * hist2))
        hist_diff = 1 - similarity

        # 2. МАКСИМАЛЬНАЯ ЛОКАЛЬНАЯ РАЗНИЦА (50%) - ключевое улучшение!
        diff_map = np.abs(ref_segment - test_segment)

        # УСИЛИВАЕМ КОНТРАСТ РАЗНИЦ
        diff_enhanced = np.clip(diff_map * 3.0, 0, 1)  # усиливаем в 3 раза!

        # Берем МАКСИМАЛЬНУЮ разницу в сегменте, а не среднюю
        diff_masked = diff_enhanced[valid_mask]
        max_diff = np.max(diff_masked) if len(diff_masked) > 0 else 0

        # 3. РАЗНИЦА СТРУКТУРЫ через лапласиан (20%)
        laplacian1 = np.abs(ndimage.laplace(ref_segment))
        laplacian2 = np.abs(ndimage.laplace(test_segment))
        laplacian_masked = np.abs(laplacian1 - laplacian2)[valid_mask]
        struct_diff = np.mean(laplacian_masked) if len(laplacian_masked) > 0 else 0

        # КОМБИНИРУЕМ с акцентом на максимальные различия
        total_diff = 0.3 * hist_diff + 0.5 * max_diff + 0.2 * struct_diff

        # УСИЛИВАЕМ РЕЗУЛЬТАТ
        amplified_diff = total_diff ** 0.5  # квадратный корень УВЕЛИЧИВАЕТ большие значения

        return min(amplified_diff, 1.0)  # ограничиваем 1.0

    def _visualize_comparison(self, ref_image, test_image, ref_corners, test_corners,
                              results, grid_size, diff_threshold):
        """Визуализирует результаты сравнения"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Эталонное изображение с углами
        axes[0, 0].imshow(ref_image)
        if ref_corners:
            x_corners = [p[0] for p in ref_corners]
            y_corners = [p[1] for p in ref_corners]
            axes[0, 0].plot(x_corners, y_corners, 'ro', markersize=8, markeredgecolor='white')
            for i in range(4):
                x1, y1 = ref_corners[i]
                x2, y2 = ref_corners[(i + 1) % 4]
                axes[0, 0].plot([x1, x2], [y1, y2], 'g-', linewidth=2)
        axes[0, 0].set_title('Эталонная плата')
        axes[0, 0].axis('off')

        # 2. Тестовая плата с углами
        axes[0, 1].imshow(test_image)
        if test_corners:
            x_corners = [p[0] for p in test_corners]
            y_corners = [p[1] for p in test_corners]
            axes[0, 1].plot(x_corners, y_corners, 'ro', markersize=8, markeredgecolor='white')
            for i in range(4):
                x1, y1 = test_corners[i]
                x2, y2 = test_corners[(i + 1) % 4]
                axes[0, 1].plot([x1, x2], [y1, y2], 'g-', linewidth=2)
        axes[0, 1].set_title('Тестируемая плата')
        axes[0, 1].axis('off')

        # 3. Сетка на эталоне
        axes[0, 2].imshow(ref_image)
        self._draw_grid(axes[0, 2], ref_corners, grid_size)
        axes[0, 2].set_title('Сетка сравнения')
        axes[0, 2].axis('off')

        # 4. Результаты сравнения
        axes[1, 0].imshow(ref_image)
        self._draw_comparison_results(axes[1, 0], results, diff_threshold)
        axes[1, 0].set_title('Результаты сравнения')
        axes[1, 0].axis('off')

        # 5. Статистика дефектов
        defect_count = sum(1 for r in results if r['is_defect'])
        total_segments = len(results)

        axes[1, 1].text(0.5, 0.6, f'Всего сегментов: {total_segments}\n'
                                  f'Дефектных: {defect_count}\n'
                                  f'Процент дефектов: {defect_count / total_segments * 100:.1f}%',
                        ha='center', va='center', fontsize=14,
                        transform=axes[1, 1].transAxes)

        # Простой график
        segments = list(range(total_segments))
        differences = [r['difference'] for r in results]

        axes[1, 1].bar(segments, differences, alpha=0.7)
        axes[1, 1].axhline(y=0.1, color='r', linestyle='--', label='Порог дефекта')
        axes[1, 1].set_xlabel('Номер сегмента')
        axes[1, 1].set_ylabel('Уровень различий')
        axes[1, 1].set_title('График различий по сегментам')
        axes[1, 1].legend()

        # 6. Детализация дефектов
        axes[1, 2].imshow(ref_image)
        self._highlight_defects(axes[1, 2], results)
        axes[1, 2].set_title('Выделение дефектных областей')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

        # Вывод статистики в консоль
        print(f"\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
        print(f"Всего сегментов: {total_segments}")
        print(f"Дефектных сегментов: {defect_count}")
        print(f"Процент дефектов: {defect_count / total_segments * 100:.1f}%")

        if defect_count > 0:
            print("\nДефектные сегменты:")
            for result in results:
                if result['is_defect']:
                    i, j = result['grid_pos']
                    print(f"  Сектор ({i}, {j}): разница = {result['difference']:.3f}")

    def _draw_grid(self, ax, corners, grid_size):
        """Рисует сетку на изображении"""
        if len(corners) != 4:
            return

        # Создаем маску для определения границ платы
        h, w = ax.images[0].get_array().shape[:2]
        mask = self.create_pcb_mask((h, w), corners)

        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return

        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)

        pcb_height = max_y - min_y
        pcb_width = max_x - min_x

        segment_h = pcb_height // grid_size
        segment_w = pcb_width // grid_size

        # Рисуем вертикальные линии
        for j in range(1, grid_size):
            x = min_x + j * segment_w
            ax.plot([x, x], [min_y, max_y], 'w-', alpha=0.3)

        # Рисуем горизонтальные линии
        for i in range(1, grid_size):
            y = min_y + i * segment_h
            ax.plot([min_x, max_x], [y, y], 'w-', alpha=0.3)

    def _draw_comparison_results(self, ax, results, diff_threshold):
        """Рисует результаты сравнения"""
        for result in results:
            x_start, y_start, x_end, y_end = result['coords']
            difference = result['difference']

            # Цвет в зависимости от уровня различий
            if difference > diff_threshold:
                color = 'red'
                alpha = 0.6
            elif difference > diff_threshold * 0.5:
                color = 'yellow'
                alpha = 0.4
            else:
                color = 'green'
                alpha = 0.3

            rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                 fill=True, color=color, alpha=alpha)
            ax.add_patch(rect)

    def _highlight_defects(self, ax, results):
        """Выделяет дефектные области"""
        for result in results:
            if result['is_defect']:
                x_start, y_start, x_end, y_end = result['coords']
                rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                     fill=True, color='red', alpha=0.5)
                ax.add_patch(rect)


def main():
    # Создаем компаратор
    comparator = PCBComparator()

    # Устанавливаем эталонное изображение
    reference_path = "фотка1.jpeg"  # путь к эталону
    comparator.set_reference(reference_path)

    # Сравниваем с тестовым изображением
    test_path = "фотка2.jpeg"  # путь к тестовому изображению
    # diff_threshold задаёт максимально допустимую разницу между сегментами (0..1)
    results = comparator.compare_with_reference(test_path, grid_size=10, diff_threshold=0.8)

    # Можно сравнить несколько изображений
    # test_path2 = "test_pcb2.jpg"
    # results2 = comparator.compare_with_reference(test_path2, grid_size=10)


if __name__ == "__main__":
    main()