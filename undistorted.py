import numpy as np
import cv2
import glob
import pickle

# Определите размеры шахматной доски
chessboard_size = (8, 5)  # (количество внутренних углов по горизонтали, количество внутренних углов по вертикали)

# create and save undistorted matrix
def save_undistorted_matrix():
    # Подготовка объектных точек
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3D точки в реальном пространстве
    imgpoints = []  # 2D точки в плоскости изображения

    # Загрузка изображений
    images = glob.glob('.\\chessboard\*.*')  # Укажите путь к вашим изображениям
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Находим углы шахматной доски
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"Углы не найдены для изображения: {fname}")

    # Проверка длины массивов
    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("Не удалось найти углы шахматной доски на всех изображениях.")
    else:
        # Используем размер последнего изображения для калибровки
        h, w = gray.shape  # Получаем размер последнего изображения
        # Калибровка камеры
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

        with open('camera_calibration.pkl', 'wb') as f:
            pickle.dump((mtx, dist), f)

# Устранение дисторсии для каждого изображения в папке
def undistorted_all_in_folder(path_to_folder_in, path_to_folder_out):
    path_to_files = path_to_folder_in + "*.*"
    images = glob.glob(path_to_files)  # Укажите путь к вашим изображениям
    with open('camera_calibration.pkl', 'rb') as f:
        mtx, dist = pickle.load(f)
    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]

        # Получаем оптимальную матрицу камеры
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # print(newcameramtx)

        # Устранение дисторсии
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # Обрезаем изображение по ROI
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # Сохраняем результат
        cv2.imwrite(path_to_folder_out + fname.split('\\')[-1], dst)


def undistort_my(path):
    # Загрузка переменных из файла
    with open('camera_calibration.pkl', 'rb') as f:
        mtx, dist = pickle.load(f)

    img = cv2.imread(path)
    h, w = img.shape[:2]

    # Получаем оптимальную матрицу камеры
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # print(newcameramtx)

    # Устранение дисторсии
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Обрезаем изображение по ROI
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    # Сохраняем результат
    cv2.imwrite('.\\undistorted\\undistorted_' + path.split('\\')[-1], dst)


def undistort_my_1(byte_array):
    # Загрузка переменных из файла
    with open('camera_calibration.pkl', 'rb') as f:
        mtx, dist = pickle.load(f)

    # Преобразование bytearray в numpy массив
    nparr = np.frombuffer(byte_array, np.uint8)

    # Декодирование изображения
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Получаем оптимальную матрицу камеры
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # print(newcameramtx)

    # Устранение дисторсии
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Обрезаем изображение по ROI
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    # Сохраняем результат
    cv2.imwrite('.\\undistorted\\undistorted_bytes.png', dst)
    # Кодирование изображения в JPEG
    success, encoded_image = cv2.imencode('.JPEG', dst)

    if success:
        # Преобразование в bytearray
        byte_array = bytearray(encoded_image)
        return byte_array


def print_cam_calib(path):
    with open(path, 'rb') as f:
        mtx, dist = pickle.load(f)
    print(mtx)
    print(dist)

# chessboard one image
# undistort_my('.\static\\received_image.png')


# chessboard all images in folder
epoch = "0"

# folder_in = "D:\projects\python\kursovaya_web_app\static\\plate_distort_" + epoch + "\\"
# folder_out = "D:\projects\python\kursovaya_web_app\static\\plate_undistort_" + epoch + "\\"
folder_in = "D:\projects\python\\big_challenge\img" + "\\"
folder_out = "D:\projects\python\\big_challenge\\undistorted" + "\\"

undistorted_all_in_folder(folder_in, folder_out)

# create chessboard matrix
# save_undistorted_matrix()


#print chessboard matrix
# # path = 'camera_calibration.pkl'
# path = 'camera_param.pkl'
# print("past:")
# print_cam_calib(path)
#
# path = 'camera_calibration.pkl'
# # path = 'camera_param.pkl'
# print("now:")
# print_cam_calib(path)
