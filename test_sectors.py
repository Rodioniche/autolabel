import dataclasses
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from matplotlib.widgets import Slider, CheckButtons, RadioButtons


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


def to_grayscale_float(image: ArrayLike) -> ArrayLike:
    if image.ndim == 3 and image.shape[2] > 1:
        return image[..., :3].mean(axis=2)
    if image.ndim == 3:
        return image[..., 0]
    return image


def rgb_to_hsv(image: ArrayLike) -> ArrayLike:
    arr = image[..., :3]
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    delta = maxc - minc + 1e-8

    v = maxc
    s = delta / (maxc + 1e-8)

    rc = (maxc - arr[..., 0]) / delta
    gc = (maxc - arr[..., 1]) / delta
    bc = (maxc - arr[..., 2]) / delta

    h = np.zeros_like(maxc)
    mask = delta > 1e-8
    r_mask = (arr[..., 0] == maxc) & mask
    g_mask = (arr[..., 1] == maxc) & mask
    b_mask = (arr[..., 2] == maxc) & mask

    h[r_mask] = (bc - gc)[r_mask]
    h[g_mask] = 2.0 + (rc - bc)[g_mask]
    h[b_mask] = 4.0 + (gc - rc)[b_mask]
    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=2)


def otsu_threshold(values: ArrayLike) -> float:
    hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    total = values.size
    if total == 0:
        return 0.5

    cumulative = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    global_mean = cumulative_mean[-1]

    numerator = (global_mean * cumulative - cumulative_mean) ** 2
    denominator = cumulative * (total - cumulative)
    denominator[denominator == 0] = 1
    sigma_b_squared = numerator / denominator
    idx = np.argmax(sigma_b_squared)
    return bin_edges[idx]


def largest_component(mask: ArrayLike) -> ArrayLike:
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask
    counts = ndi.sum(mask, labeled, index=np.arange(1, num + 1))
    largest = np.argmax(counts) + 1
    return labeled == largest


@dataclasses.dataclass
class SegmentationParams:
    mix_ratio: float = 0.0  # 0 = чистый Value, 1 = чистая Saturation
    invert: bool = False
    gaussian_sigma: float = 2.0
    opening_size: int = 5
    closing_size: int = 7
    polarity: str = "auto"  # "auto", "high", "low"


def segment_board(image: ArrayLike, params: SegmentationParams | None = None, debug: bool = False) -> ArrayLike:
    params = params or SegmentationParams()
    hsv = rgb_to_hsv(image)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    combined = (1.0 - params.mix_ratio) * value + params.mix_ratio * saturation
    if params.invert:
        combined = 1.0 - combined
    blurred = ndi.gaussian_filter(combined, sigma=params.gaussian_sigma)
    threshold = otsu_threshold(blurred)

    mask_high = blurred >= threshold
    mask_low = blurred < threshold

    if params.polarity == "auto":
        ratio = mask_high.mean()
        mask = mask_high if 0.05 < ratio < 0.95 else mask_low
    elif params.polarity == "high":
        mask = mask_high
    else:
        mask = mask_low

    opening_kernel = np.ones((params.opening_size, params.opening_size))
    closing_kernel = np.ones((params.closing_size, params.closing_size))
    mask = ndi.binary_opening(mask, structure=opening_kernel)
    mask = ndi.binary_closing(mask, structure=closing_kernel)
    mask = largest_component(mask)
    mask = ndi.binary_fill_holes(mask)

    if debug:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("RGB")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(hsv[..., 0], cmap="hsv")
        axes[0, 1].set_title("Hue")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(saturation, cmap="gray")
        axes[0, 2].set_title("Saturation")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(value, cmap="gray")
        axes[1, 0].set_title("Value")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(blurred, cmap="inferno")
        axes[1, 1].set_title(f"Combined + blur (thr={threshold:.3f})")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(mask, cmap="gray")
        axes[1, 2].set_title("Mask")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.hist(blurred.ravel(), bins=50, color="gray", alpha=0.8)
        plt.axvline(threshold, color="red", linestyle="--", label="Otsu threshold")
        plt.title("Histogram of combined channel")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"mask_ratio={mask.mean():.3f}, polarity={params.polarity}")

    return mask


def tune_segmentation(image: ArrayLike, initial: SegmentationParams | None = None) -> SegmentationParams:
    params = initial or SegmentationParams()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25)

    hsv = rgb_to_hsv(image)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    imgs = {
        "rgb": axes[0, 0].imshow(image),
        "hue": axes[0, 1].imshow(hsv[..., 0], cmap="hsv"),
        "sat": axes[0, 2].imshow(saturation, cmap="gray"),
        "val": axes[1, 0].imshow(value, cmap="gray"),
        "combined": axes[1, 1].imshow(value, cmap="inferno"),
        "mask": axes[1, 2].imshow(np.zeros_like(value), cmap="gray", vmin=0, vmax=1),
    }
    axes[0, 0].set_title("RGB"); axes[0, 1].set_title("Hue"); axes[0, 2].set_title("Sat")
    axes[1, 0].set_title("Value"); axes[1, 1].set_title("Combined"); axes[1, 2].set_title("Mask")
    for ax in axes.ravel():
        ax.axis("off")

    axcolor = "lightgoldenrodyellow"
    slider_height = 0.02
    sliders = [
        ("mix", 0.0, 1.0, params.mix_ratio, "Смесь (0=Value,1=Sat)"),
        ("sigma", 0.1, 5.0, params.gaussian_sigma, "Sigma"),
        ("open", 1, 15, params.opening_size, "Opening"),
        ("close", 1, 21, params.closing_size, "Closing"),
    ]

    slider_objs = {}
    for idx, (name, vmin, vmax, val, label) in enumerate(sliders):
        ax = plt.axes([0.1, 0.15 - idx * slider_height, 0.65, slider_height], facecolor=axcolor)
        slider = Slider(ax, label, vmin, vmax, valinit=val, valstep=None)
        slider_objs[name] = slider

    check_ax = plt.axes([0.8, 0.1, 0.15, 0.08])
    invert_check = CheckButtons(check_ax, ["Invert"], [params.invert])

    radio_ax = plt.axes([0.8, 0.02, 0.15, 0.08])
    polarity_radio = RadioButtons(radio_ax, ("auto", "high", "low"), active=("auto", "high", "low").index(params.polarity))

    hist_fig, hist_ax = plt.subplots(figsize=(6, 3))

    def recompute(_=None):
        opening = max(1, int(round(slider_objs["open"].val)))
        closing = max(1, int(round(slider_objs["close"].val)))
        if opening % 2 == 0:
            opening += 1
        if closing % 2 == 0:
            closing += 1
        current = SegmentationParams(
            mix_ratio=float(slider_objs["mix"].val),
            invert=invert_check.get_status()[0],
            gaussian_sigma=float(slider_objs["sigma"].val),
            opening_size=opening,
            closing_size=closing,
            polarity=polarity_radio.value_selected,
        )
        mask = segment_board(image, current, debug=False)

        combined = (1.0 - current.mix_ratio) * value + current.mix_ratio * saturation
        if current.invert:
            combined = 1.0 - combined
        blurred = ndi.gaussian_filter(combined, sigma=current.gaussian_sigma)
        threshold = otsu_threshold(blurred)

        imgs["combined"].set_data(blurred)
        imgs["mask"].set_data(mask.astype(float))
        axes[1, 1].set_title(f"Combined (thr {threshold:.3f})")

        hist_ax.clear()
        hist_ax.hist(blurred.ravel(), bins=50, color="gray", alpha=0.8)
        hist_ax.axvline(threshold, color="red", linestyle="--")
        hist_ax.set_title("Histogram")
        hist_ax.figure.canvas.draw_idle()

        fig.canvas.draw_idle()
        return current

    for slider in slider_objs.values():
        slider.on_changed(recompute)
    invert_check.on_clicked(recompute)
    polarity_radio.on_clicked(recompute)

    recompute()
    plt.show(block=True)

    return SegmentationParams(
        mix_ratio=float(slider_objs["mix"].val),
        invert=invert_check.get_status()[0],
        gaussian_sigma=float(slider_objs["sigma"].val),
        opening_size=max(1, int(round(slider_objs["open"].val))) | 1,
        closing_size=max(1, int(round(slider_objs["close"].val))) | 1,
        polarity=polarity_radio.value_selected,
    )


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


def match_histogram(
    reference: ArrayLike,
    source: ArrayLike,
    mask: Optional[ArrayLike] = None,
    bins: int = 256,
) -> ArrayLike:
    """Подгоняет гистограмму источника к эталону поканально."""
    result = source.copy()

    def _channel_match(ref_channel: ArrayLike, src_channel: ArrayLike) -> ArrayLike:
        ref_vals = ref_channel.ravel()
        src_vals = src_channel.ravel()
        if mask is not None:
            mask_flat = mask.ravel()
            ref_vals = ref_vals[mask_flat]

        ref_vals = np.clip(ref_vals, 0.0, 1.0)
        src_vals = np.clip(src_vals, 0.0, 1.0)

        ref_hist, ref_bins = np.histogram(ref_vals, bins=bins, range=(0.0, 1.0))
        src_hist, src_bins = np.histogram(src_vals, bins=bins, range=(0.0, 1.0))

        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        if ref_cdf[-1] == 0 or src_cdf[-1] == 0:
            return src_channel

        ref_cdf /= ref_cdf[-1]
        src_cdf /= src_cdf[-1]

        ref_bin_centers = (ref_bins[:-1] + ref_bins[1:]) / 2.0

        mapping = np.interp(src_cdf, ref_cdf, ref_bin_centers)
        src_indices = np.clip(((src_channel * (bins - 1))).astype(int), 0, bins - 1)
        matched = mapping[src_indices]
        return np.clip(matched, 0.0, 1.0)

    if result.ndim == 3:
        for c in range(result.shape[2]):
            ref_ch = reference[..., c]
            src_ch = result[..., c]
            result[..., c] = _channel_match(ref_ch, src_ch)
    else:
        result = _channel_match(reference, result)

    return result


def to_grayscale_float(image: ArrayLike) -> ArrayLike:
    if image.ndim == 3 and image.shape[2] > 1:
        return image[..., :3].mean(axis=2)
    if image.ndim == 3:
        return image[..., 0]
    return image


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


@dataclasses.dataclass
class GoodFeaturesParams:
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: float = 10.0
    block_size: int = 5
    use_harris: bool = True
    harris_k: float = 0.04


class GoodFeaturesDetector:
    def __init__(self, params: GoodFeaturesParams | None = None):
        self.params = params or GoodFeaturesParams()

    def detect(self, image: ArrayLike, mask: Optional[ArrayLike] = None) -> list[tuple[float, float, float]]:
        gray = to_grayscale_float(image)
        block = max(3, self.params.block_size | 1)

        Ix = ndi.sobel(gray, axis=1)
        Iy = ndi.sobel(gray, axis=0)
        Ixx = ndi.uniform_filter(Ix * Ix, size=block)
        Iyy = ndi.uniform_filter(Iy * Iy, size=block)
        Ixy = ndi.uniform_filter(Ix * Iy, size=block)

        if self.params.use_harris:
            det = Ixx * Iyy - Ixy * Ixy
            trace = Ixx + Iyy
            response = det - self.params.harris_k * (trace ** 2)
        else:
            trace = Ixx + Iyy
            temp = np.sqrt((Ixx - Iyy) ** 2 + 4 * (Ixy ** 2))
            lambda1 = (trace + temp) / 2.0
            lambda2 = (trace - temp) / 2.0
            response = np.minimum(lambda1, lambda2)

        response[response < 0] = 0
        max_response = response.max()
        if max_response <= 0:
            return []

        threshold = self.params.quality_level * max_response
        candidates = np.argwhere(response >= threshold)
        if candidates.size == 0:
            return []

        scores = response[response >= threshold]
        order = np.argsort(scores)[::-1]
        candidates = candidates[order]
        scores = scores[order]

        half_block = block // 2
        height, width = gray.shape
        selected: list[tuple[float, float, float]] = []

        if mask is not None:
            if mask.shape != gray.shape:
                raise ValueError("Маска должна совпадать размером с изображением")
            mask_bool = mask.astype(bool)
        else:
            mask_bool = None

        for (y, x), score in zip(candidates, scores):
            if x < half_block or x >= width - half_block:
                continue
            if y < half_block or y >= height - half_block:
                continue
            if mask_bool is not None and not mask_bool[y, x]:
                continue

            keep = True
            for _, px, py in selected:
                if (x - px) ** 2 + (y - py) ** 2 < self.params.min_distance ** 2:
                    keep = False
                    break
            if not keep:
                continue

            selected.append((float(score), float(x), float(y)))
            if len(selected) >= self.params.max_corners:
                break

        return selected


def select_board_corners(points: list[tuple[float, float, float]], mask: ArrayLike) -> PointSeq:
    if not points:
        return []

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))

    quadrants = {"tl": None, "tr": None, "br": None, "bl": None}
    for score, x, y in points:
        if x < center_x and y < center_y:
            key = "tl"
        elif x >= center_x and y < center_y:
            key = "tr"
        elif x >= center_x and y >= center_y:
            key = "br"
        else:
            key = "bl"

        current = quadrants[key]
        if current is None or score > current[0]:
            quadrants[key] = (score, x, y)

    result = []
    for key in ("tl", "tr", "br", "bl"):
        if quadrants[key] is None:
            return []
        _, x, y = quadrants[key]
        result.append((x, y))
    return result


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

    @staticmethod
    def from_manual_selection(
        reference_image: ArrayLike,
        test_image: ArrayLike,
        message: str = "Выберите 4 угла платы по часовой стрелке",
    ) -> "AlignmentConfig":
        def _collect_points(image: ArrayLike, title: str) -> list[tuple[float, float]]:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image)
            ax.set_title(f"{title}\n{message}")
            ax.axis("off")
            plt.tight_layout()
            points = plt.ginput(4, timeout=0)
            plt.close(fig)
            if len(points) != 4:
                raise RuntimeError("Ожидалось 4 клика для определения углов.")
            return points

        print("Отметьте углы на эталонном изображении.")
        ref_pts = _collect_points(reference_image, "Эталон")
        print("Отметьте углы на тестовом изображении.")
        test_pts = _collect_points(test_image, "Тест")
        print("Эталонные углы:", ref_pts)
        print("Тестовые углы:", test_pts)
        return AlignmentConfig(reference_corners=ref_pts, test_corners=test_pts)


def demo(reference_path: str, test_path: str) -> None:
    """Простейший прогон: загружаем, нормализуем и показываем результат."""
    ref = load_image(reference_path)
    test = load_image(test_path)

    preprocessor = PCBPreprocessor()
    ref_proc = preprocessor.preprocess(ref)
    test_proc = preprocessor.preprocess(test)
    test_proc = match_global_intensity(ref_proc, test_proc)
    test_proc = match_histogram(ref_proc, test_proc)

    seg_params = SegmentationParams(
        mix_ratio=0.0,      # 0 = чистый Value
        invert=False,       # True, если плата темнее фона
        gaussian_sigma=2.0,
        opening_size=5,
        closing_size=7,
        polarity="auto",    # "auto" / "high" / "low"
    )
    interactive_segmentation = False
    if interactive_segmentation:
        print("Запуск интерактивного тюнера сегментации для эталона.")
        seg_params = tune_segmentation(ref_proc, seg_params)
    debug_segmentation = not interactive_segmentation
    ref_mask = segment_board(ref_proc, seg_params, debug=debug_segmentation)
    test_mask = segment_board(test_proc, seg_params, debug=debug_segmentation)

    detector = GoodFeaturesDetector(
        GoodFeaturesParams(
            max_corners=300,
            quality_level=0.01,
            min_distance=15.0,
            block_size=7,
            use_harris=True,
            harris_k=0.04,
        )
    )

    ref_points = detector.detect(ref_proc, mask=ref_mask)
    test_points = detector.detect(test_proc, mask=test_mask)
    ref_corners = select_board_corners(ref_points, ref_mask)
    test_corners = select_board_corners(test_points, test_mask)

    def show_corners(image: ArrayLike, corners: PointSeq, title: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        if corners:
            xs = [p[0] for p in corners] + [corners[0][0]]
            ys = [p[1] for p in corners] + [corners[0][1]]
            ax.plot(xs, ys, "r-o", linewidth=2, markersize=4)
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def show_mask(image: ArrayLike, mask: ArrayLike, title: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.imshow(mask, cmap="Reds", alpha=0.3)
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    show_mask(ref_proc, ref_mask, "Эталонная маска платы")
    show_mask(test_proc, test_mask, "Тестовая маска платы")

    if len(ref_corners) == 4 and len(test_corners) == 4:
        alignment = AlignmentConfig(reference_corners=ref_corners, test_corners=test_corners)
        print("Автоматические углы (ref):", ref_corners)
        print("Автоматические углы (test):", test_corners)
        show_corners(ref_proc, ref_corners, "Эталон с найденными углами")
        show_corners(test_proc, test_corners, "Тест с найденными углами")
    else:
        print("Автоматическое определение углов не удалось, переключаемся на ручной режим.")
        alignment = AlignmentConfig.from_manual_selection(ref_proc, test_proc)
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
    demo("chess.jpg", "chessboard_3_3.jpg")
