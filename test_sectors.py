import dataclasses
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ArrayLike = np.ndarray
PointSeq = Sequence[Tuple[float, float]]


def load_image(path: str | Path, force_rgb: bool = True) -> ArrayLike:
    """Загружает изображение и возвращает массив float32 в диапазоне [0, 1]."""
    with Image.open(path) as img:
        if force_rgb:
            img = img.convert("RGB")
        array = np.asarray(img).astype(np.float32)
    if array.ndim == 2:
        array = array[..., None]
    return array / 255.0


def save_image(path: str | Path, array: ArrayLike) -> None:
    """Сохраняет массив [0,1] обратно в файл для дебага."""
    array_uint8 = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array_uint8.shape[2] == 1:
        array_uint8 = array_uint8[..., 0]
    Image.fromarray(array_uint8).save(path)


def compute_valid_mask(image: ArrayLike, threshold: float = 0.01) -> ArrayLike:
    """Формирует маску значимых пикселей (отсекаем чистый фон)."""
    if image.ndim == 3 and image.shape[2] > 1:
        intensity = image.mean(axis=2)
    else:
        intensity = image[..., 0] if image.ndim == 3 else image
    return intensity > threshold


def stretch_dynamic_range(
    image: ArrayLike,
    mask: Optional[ArrayLike] = None,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> ArrayLike:
    """Линейно растягивает динамический диапазон по каналам."""
    stretched = image.copy()
    channels = stretched.shape[2] if stretched.ndim == 3 else 1
    mask = mask if mask is not None else compute_valid_mask(stretched)

    for c in range(channels):
        channel = stretched[..., c] if channels > 1 else stretched
        valid_pixels = channel[mask]
        if valid_pixels.size == 0:
            continue
        low = np.percentile(valid_pixels, low_percentile)
        high = np.percentile(valid_pixels, high_percentile)
        if high - low < 1e-5:
            continue
        normalized = (channel - low) / (high - low)
        if channels > 1:
            stretched[..., c] = np.clip(normalized, 0.0, 1.0)
        else:
            stretched = np.clip(normalized, 0.0, 1.0)
    return stretched


def balance_channels(image: ArrayLike, mask: Optional[ArrayLike] = None) -> ArrayLike:
    """Выравнивает среднюю яркость каналов (простая белая балансировка)."""
    balanced = image.copy()
    channels = balanced.shape[2] if balanced.ndim == 3 else 1
    mask = mask if mask is not None else compute_valid_mask(balanced)

    if channels == 1:
        return balanced

    means = []
    for c in range(channels):
        channel = balanced[..., c]
        valid_pixels = channel[mask]
        means.append(np.mean(valid_pixels) if valid_pixels.size else 0.0)

    global_mean = np.mean(means)
    if global_mean < 1e-6:
        return balanced

    for c in range(channels):
        if means[c] < 1e-6:
            continue
        scale = global_mean / means[c]
        balanced[..., c] = np.clip(balanced[..., c] * scale, 0.0, 1.0)
    return balanced


def match_global_intensity(
    target: ArrayLike,
    source: ArrayLike,
    mask: Optional[ArrayLike] = None,
) -> ArrayLike:
    """Подгоняет среднее и стандартное отклонение источника к целевому."""
    adjusted = source.copy()
    mask = mask if mask is not None else compute_valid_mask(adjusted)

    def calc_stats(arr: ArrayLike) -> Tuple[float, float]:
        values = arr[mask]
        if values.size == 0:
            return 0.5, 0.1
        mean = float(np.mean(values))
        std = float(np.std(values) + 1e-6)
        return mean, std

    if target.ndim == 3 and target.shape[2] > 1:
        target_gray = target.mean(axis=2)
    else:
        target_gray = target[..., 0] if target.ndim == 3 else target

    if adjusted.ndim == 3 and adjusted.shape[2] > 1:
        source_gray = adjusted.mean(axis=2)
    else:
        source_gray = adjusted[..., 0] if adjusted.ndim == 3 else adjusted

    tgt_mean, tgt_std = calc_stats(target_gray)
    src_mean, src_std = calc_stats(source_gray)

    if src_std < 1e-6:
        return adjusted

    scale = tgt_std / src_std
    shift = tgt_mean - src_mean * scale
    adjusted = np.clip(adjusted * scale + shift, 0.0, 1.0)
    return adjusted


@dataclasses.dataclass
class PCBPreprocessor:
    """Пошаговые преобразования снимков плат без OpenCV."""

    low_percentile: float = 1.0
    high_percentile: float = 99.0

    def preprocess(self, image: ArrayLike) -> ArrayLike:
        mask = compute_valid_mask(image)
        stretched = stretch_dynamic_range(
            image,
            mask=mask,
            low_percentile=self.low_percentile,
            high_percentile=self.high_percentile,
        )
        balanced = balance_channels(stretched, mask=mask)
        return balanced


def compute_homography(src_points: PointSeq, dst_points: PointSeq) -> ArrayLike:
    """Вычисляет 3x3 матрицу гомографии из соответствий точек."""
    if len(src_points) != 4 or len(dst_points) != 4:
        raise ValueError("Нужно четыре пары точек для гомографии.")

    A = []
    for (x, y), (u, v) in zip(src_points, dst_points):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

    _, _, vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def warp_with_homography(
    image: ArrayLike,
    H: ArrayLike,
    output_shape: Tuple[int, int],
) -> Tuple[ArrayLike, ArrayLike]:
    """Переносит изображение в новую плоскость методом ближайшего соседа."""
    height, width = output_shape
    channels = image.shape[2] if image.ndim == 3 else 1
    warped = np.zeros((height, width, channels), dtype=image.dtype)

    y_ref, x_ref = np.indices((height, width))
    homog = np.stack(
        (x_ref.ravel(), y_ref.ravel(), np.ones_like(x_ref).ravel()),
        axis=0,
    )
    src_coords = np.linalg.inv(H) @ homog
    src_x = src_coords[0] / src_coords[2]
    src_y = src_coords[1] / src_coords[2]

    src_x_int = np.round(src_x).astype(int)
    src_y_int = np.round(src_y).astype(int)

    mask_flat = (
        (src_x_int >= 0)
        & (src_x_int < image.shape[1])
        & (src_y_int >= 0)
        & (src_y_int < image.shape[0])
    )
    mask = mask_flat.reshape(height, width)

    valid_y = y_ref.ravel()[mask_flat]
    valid_x = x_ref.ravel()[mask_flat]
    src_y_valid = src_y_int[mask_flat]
    src_x_valid = src_x_int[mask_flat]

    if channels == 1:
        warped[..., 0][valid_y, valid_x] = image[src_y_valid, src_x_valid]
    else:
        warped[valid_y, valid_x, :] = image[src_y_valid, src_x_valid, :]

    return warped if channels > 1 else warped[..., 0:1], mask


def default_rect_corners(width: int, height: int) -> list[tuple[float, float]]:
    """Возвращает вершины прямоугольника по периметру изображения."""
    return [
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    ]


@dataclasses.dataclass
class AlignmentConfig:
    reference_corners: PointSeq
    test_corners: PointSeq

    @staticmethod
    def full_frame(ref_shape: Tuple[int, int], test_shape: Tuple[int, int]) -> "AlignmentConfig":
        ref_h, ref_w = ref_shape
        test_h, test_w = test_shape
        return AlignmentConfig(
            reference_corners=default_rect_corners(ref_w, ref_h),
            test_corners=default_rect_corners(test_w, test_h),
        )


def demo(reference_path: str, test_path: str) -> None:
    """Простейший прогон: загружаем, нормализуем и показываем результат."""
    ref = load_image(reference_path)
    test = load_image(test_path)

    preprocessor = PCBPreprocessor()
    ref_proc = preprocessor.preprocess(ref)
    test_proc = preprocessor.preprocess(test)
    test_proc = match_global_intensity(ref_proc, test_proc)

    alignment = AlignmentConfig.full_frame(
        ref_proc.shape[:2],
        test_proc.shape[:2],
    )
    H = compute_homography(alignment.test_corners, alignment.reference_corners)
    warped_test, warped_mask = warp_with_homography(
        test_proc,
        H,
        output_shape=ref_proc.shape[:2],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(ref_proc)
    axes[0].set_title("Эталон (норм.)")
    axes[0].axis("off")

    axes[1].imshow(warped_test)
    axes[1].set_title("Тест (норм. + гомография)")
    axes[1].axis("off")

    fig_diff, diff_ax = plt.subplots(1, 1, figsize=(6, 6))
    diff = np.abs(ref_proc - warped_test) * warped_mask[..., None]
    diff_ax.imshow(diff.mean(axis=2), cmap="inferno")
    diff_ax.set_title("Абсолютная разница (пока без порогов)")
    diff_ax.axis("off")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo("photoetalon.jpg", "photoschakal.jpg")
