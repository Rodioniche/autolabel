"""
Простой скрипт для устранения дисторсии с одного изображения
Использование: python3 fix_distortion.py your_photo.jpg
"""
import sys
from pathlib import Path

def fix_distortion(input_image: str):
    """Устраняет дисторсию с изображения."""
    
    # Проверка OpenCV
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import pickle
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Установите зависимости: pip install opencv-python numpy pillow")
        return False
    
    input_path = Path(input_image)
    if not input_path.exists():
        print(f"❌ Файл не найден: {input_path}")
        return False
    
    # Проверка файла калибровки
    calib_file = Path("camera_calibration.pkl")
    if not calib_file.exists():
        print(f"❌ Файл калибровки не найден: {calib_file.absolute()}")
        print("\n📝 Как создать файл калибровки:")
        print("1. Создайте папку 'chessboard/'")
        print("2. Сделайте 10-15 фото шахматной доски (8x5 внутренних углов)")
        print("3. Отредактируйте undistorted.py:")
        print("   - Закомментируйте строку 145: # undistorted_all_in_folder(...)")
        print("   - Раскомментируйте строку 148: save_undistorted_matrix()")
        print("4. Запустите: python3 undistorted.py")
        return False
    
    output_path = input_path.parent / f"undistorted_{input_path.name}"
    
    print("="*60)
    print("УСТРАНЕНИЕ ДИСТОРСИИ")
    print("="*60)
    print(f"\n📁 Входной файл:  {input_path}")
    print(f"📁 Выходной файл: {output_path}")
    
    try:
        # Загрузка калибровки
        print("\n1️⃣ Загрузка калибровки...")
        with open(calib_file, 'rb') as f:
            mtx, dist = pickle.load(f)
        print("   ✓ Калибровка загружена")
        
        # Загрузка изображения
        print("\n2️⃣ Загрузка изображения...")
        img = cv2.imread(str(input_path))
        if img is None:
            print("   ❌ Не удалось загрузить изображение")
            return False
        h, w = img.shape[:2]
        print(f"   ✓ Размер: {w}×{h} пикселей")
        
        # Устранение дисторсии
        print("\n3️⃣ Устранение дисторсии...")
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Обрезаем по ROI
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
        print(f"   ✓ Новый размер: {w_roi}×{h_roi} пикселей")
        
        # Сохранение
        print("\n4️⃣ Сохранение результата...")
        cv2.imwrite(str(output_path), dst)
        
        size_before = input_path.stat().st_size / 1024
        size_after = output_path.stat().st_size / 1024
        print(f"   ✓ Сохранено: {output_path}")
        print(f"   📊 Размер до:  {size_before:.1f} КБ")
        print(f"   📊 Размер после: {size_after:.1f} КБ")
        
        print("\n" + "="*60)
        print("✅ ГОТОВО!")
        print("="*60)
        print(f"\n📷 Откройте файл: {output_path.name}")
        print("   Сравните с оригиналом - дисторсия исправлена!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    
    if len(sys.argv) < 2:
        print("="*60)
        print("УСТРАНЕНИЕ ДИСТОРСИИ - Справка")
        print("="*60)
        print("\n📖 Использование:")
        print("   python3 fix_distortion.py <путь_к_фото>")
        print("\n💡 Примеры:")
        print("   python3 fix_distortion.py chess.jpg")
        print("   python3 fix_distortion.py my_photo.jpg")
        print("   python3 fix_distortion.py /path/to/image.jpg")
        print("\n📁 Результат сохранится как: undistorted_<имя_файла>.jpg")
        print("\n⚠️  Требования:")
        print("   • OpenCV: pip install opencv-python")
        print("   • Файл калибровки: camera_calibration.pkl")
        print()
        sys.exit(0)
    
    input_image = sys.argv[1]
    success = fix_distortion(input_image)
    
    if not success:
        sys.exit(1)

