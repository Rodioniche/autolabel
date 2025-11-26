"""
Тестовый скрипт для проверки нового алгоритма поиска углов платы
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Импортируем детектор из new_detect_1
from new_detect_1 import PCBCornerDetector

def test_corner_detection(image_path):
    """Тестирует поиск углов на одном изображении"""
    
    print("="*60)
    print(f"ТЕСТИРОВАНИЕ ПОИСКА УГЛОВ: {image_path}")
    print("="*60)
    
    # Загружаем изображение
    img = Image.open(image_path)
    img_array = np.array(img)
    
    print(f"Размер изображения: {img_array.shape}")
    
    # Создаем детектор
    detector = PCBCornerDetector()
    
    # Находим углы
    print("\nПоиск углов...")
    corners = detector.find_pcb_corners_simple(img_array)
    
    print(f"\nНайденные углы:")
    for i, corner in enumerate(corners):
        print(f"  Угол {i+1}: {corner}")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.imshow(img_array)
    
    if corners and len(corners) == 4:
        # Рисуем углы
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        plt.scatter(xs, ys, c='red', s=200, marker='x', linewidth=3, label='Углы')
        
        # Соединяем линиями
        corners_loop = corners + [corners[0]]
        for i in range(len(corners)):
            x1, y1 = corners_loop[i]
            x2, y2 = corners_loop[i+1]
            plt.plot([x1, x2], [y1, y2], 'g-', linewidth=3)
        
        plt.legend()
        plt.title(f'Найденные углы платы\n{image_path}')
    else:
        plt.title(f'Углы не найдены!\n{image_path}')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Тест завершён")
    print("="*60)
    return corners


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Тестирование с файлами по умолчанию...")
        print()
        
        # Тестируем оба изображения
        for img_path in ['bebebe.jpeg', 'test.png']:
            try:
                test_corner_detection(img_path)
                print()
            except FileNotFoundError:
                print(f"⚠️  Файл {img_path} не найден, пропускаем...")
                print()
            except Exception as e:
                print(f"❌ Ошибка при обработке {img_path}: {e}")
                print()
    else:
        # Тестируем указанный файл
        image_path = sys.argv[1]
        test_corner_detection(image_path)

