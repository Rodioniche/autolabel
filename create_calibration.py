"""
Скрипт для создания файла калибровки камеры из фотографий шахматной доски.
Использование: python3 create_calibration.py
"""
import sys
from pathlib import Path

def create_calibration_from_photos(
    photos_folder: str = "chessboard",
    chessboard_size: tuple = (8, 5),
    output_file: str = "camera_calibration.pkl"
):
    """
    Создает файл калибровки камеры из фотографий шахматной доски.
    
    Args:
        photos_folder: папка с фотографиями (.jpg)
        chessboard_size: размер доски (количество внутренних углов по горизонтали, вертикали)
        output_file: имя файла для сохранения калибровки
    """
    
    # Проверка OpenCV
    try:
        import cv2
        import numpy as np
        import pickle
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Установите зависимости: pip install opencv-python numpy")
        return False
    
    photos_path = Path(photos_folder)
    
    print("="*60)
    print("СОЗДАНИЕ ФАЙЛА КАЛИБРОВКИ КАМЕРЫ")
    print("="*60)
    print(f"\n📁 Папка с фото: {photos_path.absolute()}")
    print(f"📐 Размер доски: {chessboard_size[0]}×{chessboard_size[1]} внутренних углов")
    print(f"💾 Выходной файл: {output_file}")
    
    # Проверка папки
    if not photos_path.exists():
        print(f"\n❌ Папка не найдена: {photos_path.absolute()}")
        print("\n📝 Создайте папку и добавьте фото:")
        print(f"   mkdir {photos_folder}")
        print(f"   # Поместите 10-15 фото шахматной доски в папку {photos_folder}/")
        return False
    
    # Поиск фотографий
    print("\n1️⃣ Поиск фотографий...")
    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
    images = []
    for ext in image_extensions:
        images.extend(photos_path.glob(ext))
    
    if not images:
        print(f"   ❌ Фотографии не найдены в {photos_path}")
        print("\n📝 Добавьте фотографии:")
        print(f"   Скопируйте 10-15 .jpg файлов в папку {photos_folder}/")
        return False
    
    print(f"   ✓ Найдено {len(images)} фотографий")
    
    # Подготовка объектных точек
    print("\n2️⃣ Подготовка к калибровке...")
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    objpoints = []  # 3D точки в реальном пространстве
    imgpoints = []  # 2D точки в плоскости изображения
    
    # Обработка фотографий
    print("\n3️⃣ Обработка фотографий...")
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(images, 1):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"   ⚠️  [{i}/{len(images)}] Не удалось загрузить: {image_path.name}")
            failed += 1
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Поиск углов шахматной доски
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            successful += 1
            print(f"   ✓ [{i}/{len(images)}] {image_path.name} - углы найдены")
        else:
            failed += 1
            print(f"   ✗ [{i}/{len(images)}] {image_path.name} - углы НЕ найдены")
    
    print(f"\n   📊 Успешно: {successful}, Неудачно: {failed}")
    
    # Проверка количества успешных изображений
    if successful < 3:
        print(f"\n❌ Недостаточно фотографий для калибровки (нужно минимум 3, найдено {successful})")
        print("\n💡 Советы:")
        print("   • Убедитесь, что на фото видна ВСЯ шахматная доска")
        print("   • Проверьте размер доски (должно быть 8×5 внутренних углов)")
        print("   • Используйте хорошее освещение")
        print("   • Доска должна быть плоской, без изгибов")
        return False
    
    if successful < 10:
        print(f"\n⚠️  Предупреждение: Лучше использовать 10-15 фотографий (найдено {successful})")
        print("   Калибровка будет менее точной")
    
    # Калибровка камеры
    print("\n4️⃣ Калибровка камеры...")
    try:
        h, w = gray.shape
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), None, None
        )
        
        if not ret:
            print("   ❌ Калибровка не удалась")
            return False
        
        print("   ✓ Калибровка выполнена успешно")
        print(f"\n   📐 Матрица камеры:\n{mtx}")
        print(f"\n   📉 Коэффициенты дисторсии:\n{dist}")
        
    except Exception as e:
        print(f"   ❌ Ошибка калибровки: {e}")
        return False
    
    # Сохранение результата
    print("\n5️⃣ Сохранение файла калибровки...")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump((mtx, dist), f)
        
        output_path = Path(output_file)
        file_size = output_path.stat().st_size
        print(f"   ✓ Файл сохранен: {output_path.absolute()}")
        print(f"   📊 Размер файла: {file_size} байт")
        
    except Exception as e:
        print(f"   ❌ Ошибка сохранения: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ КАЛИБРОВКА ЗАВЕРШЕНА!")
    print("="*60)
    print(f"\n📁 Файл калибровки создан: {output_file}")
    print(f"📸 Использовано фотографий: {successful}")
    print("\n🎯 Теперь можно использовать:")
    print("   • python3 fix_distortion.py your_photo.jpg")
    print("   • python3 new_detect_1.py")
    
    return True


if __name__ == "__main__":
    print()
    
    # Проверка аргументов
    photos_folder = sys.argv[1] if len(sys.argv) > 1 else "chessboard"
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("="*60)
        print("СОЗДАНИЕ ФАЙЛА КАЛИБРОВКИ - Справка")
        print("="*60)
        print("\n📖 Использование:")
        print("   python3 create_calibration.py [папка_с_фото]")
        print("\n💡 Примеры:")
        print("   python3 create_calibration.py")
        print("   python3 create_calibration.py chessboard")
        print("   python3 create_calibration.py my_photos")
        print("\n📋 Требования:")
        print("   1. Создайте папку (по умолчанию: chessboard/)")
        print("   2. Поместите туда 10-15 фото шахматной доски (.jpg)")
        print("   3. Размер доски: 8×5 внутренних углов (9×6 клеток)")
        print("   4. Установите OpenCV: pip install opencv-python")
        print("\n📐 Размер доски:")
        print("   • 8 внутренних углов по горизонтали")
        print("   • 5 внутренних углов по вертикали")
        print("   • Это стандартная шахматная доска для калибровки")
        print("\n💡 Советы по съемке:")
        print("   • Фотографируйте доску с разных углов")
        print("   • Используйте хорошее освещение")
        print("   • Доска должна быть плоской")
        print("   • Вся доска должна быть видна на фото")
        print()
        sys.exit(0)
    
    success = create_calibration_from_photos(photos_folder)
    
    if not success:
        print("\n⚠️  Калибровка не выполнена")
        print("Запустите с --help для справки")
        sys.exit(1)

