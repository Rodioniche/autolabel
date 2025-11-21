import dataclasses
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from matplotlib.path import Path as MplPath
try:
    from skimage import color, exposure, feature, transform  # type: ignore[import]
    from skimage.measure import ransac  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError("Требуется пакет scikit-image: pip install scikit-image") from exc

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


def find_equals(src_points: Array, dst_points: Array) -> Array:
    """
    Вычисляет матрицу гомографии H (3x3) из соответствий точек.
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError("Нужно минимум 4 точки для вычисления гомографии")
    if len(src_points) != len(dst_points):
        raise ValueError("Количество исходных и целевых точек должно совпадать")

    A = []
    for (x, y), (x_prime, y_prime) in zip(src_points, dst_points):
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y, -y_prime])
    A = np.asarray(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = H / H[2, 2]
    return H


def extract_pcb_region(image: Array, corners: List[Tuple[float, float]], padding: int = 5
                       ) -> Tuple[Array, Tuple[int, int, int, int], Array]:
    """
    Вырезает область платы внутри указанных углов, дополнительно расширяя маску padding-пикселями.
    """
    if not corners or len(corners) < 4:
        h, w = image.shape[:2]
        return image.copy(), (0, 0, w, h), np.ones((h, w), dtype=bool)

    h, w = image.shape[:2]
    poly_path = MplPath(corners)
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    coords = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    mask = poly_path.contains_points(coords).reshape(h, w)

    if padding > 0:
        mask = ndi.binary_dilation(mask, iterations=padding)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return image.copy(), (0, 0, w, h), mask

    min_y = max(0, ys.min())
    max_y = min(h, ys.max() + 1)
    min_x = max(0, xs.min())
    max_x = min(w, xs.max() + 1)

    clipped_mask = mask[min_y:max_y, min_x:max_x]
    masked_img = image.copy()
    masked_img[~mask] = 0
    extracted = masked_img[min_y:max_y, min_x:max_x]

    return extracted, (min_x, min_y, max_x, max_y), clipped_mask


def adapt_homography_for_cropped_regions(
    H: Array,
    ref_bbox: Tuple[int, int, int, int],
    test_bbox: Tuple[int, int, int, int],
) -> Array:
    """
    Смещает гомографию к локальным координатам вырезанных областей.
    """
    ref_min_x, ref_min_y, _, _ = ref_bbox
    test_min_x, test_min_y, _, _ = test_bbox

    T_ref = np.array([[1, 0, -ref_min_x],
                      [0, 1, -ref_min_y],
                      [0, 0, 1]])
    T_test_inv = np.array([[1, 0, test_min_x],
                           [0, 1, test_min_y],
                           [0, 0, 1]])

    return T_ref @ H @ T_test_inv


def shift_corners_to_origin(corners: List[Tuple[float, float]], bbox: Tuple[int, int, int, int]
                            ) -> List[Tuple[float, float]]:
    """
    Переводит координаты углов в локальную систему (0,0) относительно bounding-box.
    """
    if not corners or len(corners) < 4:
        return corners
    min_x, min_y = bbox[0], bbox[1]
    return [(x - min_x, y - min_y) for (x, y) in corners]


def transform_pcb_to_reference(
    test_pcb_region: Array,
    H_adapted: Array,
    reference_pcb_region_shape: Tuple[int, ...],
) -> Array:
    """
    Преобразует вырезанную область тестовой платы в координаты вырезанной области эталонной платы.
    """
    ref_height, ref_width = reference_pcb_region_shape[:2]
    if len(reference_pcb_region_shape) == 3:
        transformed = np.zeros((ref_height, ref_width, reference_pcb_region_shape[2]),
                               dtype=test_pcb_region.dtype)
    else:
        transformed = np.zeros((ref_height, ref_width), dtype=test_pcb_region.dtype)

    y_ref, x_ref = np.indices((ref_height, ref_width))
    ones = np.ones_like(x_ref)
    ref_coords_homogeneous = np.stack([x_ref.ravel(), y_ref.ravel(), ones.ravel()])

    H_adapted_inv = np.linalg.inv(H_adapted)
    test_coords_homogeneous = H_adapted_inv @ ref_coords_homogeneous
    test_x = test_coords_homogeneous[0] / test_coords_homogeneous[2]
    test_y = test_coords_homogeneous[1] / test_coords_homogeneous[2]

    test_x_int = np.round(test_x).astype(int)
    test_y_int = np.round(test_y).astype(int)
    valid_mask = (
        (test_x_int >= 0)
        & (test_x_int < test_pcb_region.shape[1])
        & (test_y_int >= 0)
        & (test_y_int < test_pcb_region.shape[0])
    )

    valid_indices_ref = np.where(valid_mask.reshape(ref_height, ref_width))
    valid_indices_test_y = test_y_int[valid_mask]
    valid_indices_test_x = test_x_int[valid_mask]

    transformed[valid_indices_ref[0], valid_indices_ref[1]] = test_pcb_region[
        valid_indices_test_y, valid_indices_test_x
    ]
    return transformed


@dataclasses.dataclass
class AlignmentResult:
    success: bool
    warped: Array
    valid_mask: Array
    inliers: int
    total_matches: int
    message: str = ""


def _prepare_for_alignment(image: Array) -> Array:
    gray = color.rgb2gray(image)
    enhanced = exposure.equalize_adapthist(gray, clip_limit=0.03)
    return enhanced.astype(np.float32)


def align_images(reference: Array, test: Array, max_keypoints: int = 2000) -> AlignmentResult:
    ref_gray = _prepare_for_alignment(reference)
    test_gray = _prepare_for_alignment(test)

    orb_ref = feature.ORB(n_keypoints=max_keypoints, fast_threshold=0.08)
    orb_ref.detect_and_extract(ref_gray)
    orb_test = feature.ORB(n_keypoints=max_keypoints, fast_threshold=0.08)
    orb_test.detect_and_extract(test_gray)

    if len(orb_ref.keypoints) < 10 or len(orb_test.keypoints) < 10:
        msg = "Недостаточно ключевых точек для гомографии."
        return AlignmentResult(False, test, np.ones(ref_gray.shape, dtype=bool), 0, 0, msg)

    matches = feature.match_descriptors(
        orb_ref.descriptors,
        orb_test.descriptors,
        cross_check=True,
        metric="hamming",
    )

    if len(matches) < 8:
        msg = "Недостаточно совпадений дескрипторов."
        return AlignmentResult(False, test, np.ones(ref_gray.shape, dtype=bool), 0, len(matches), msg)

    src = orb_test.keypoints[matches[:, 1]]
    dst = orb_ref.keypoints[matches[:, 0]]

    model, inliers = ransac(
        (src, dst),
        transform.ProjectiveTransform,
        min_samples=4,
        residual_threshold=2.0,
        max_trials=5000,
    )

    if model is None or inliers is None or inliers.sum() < 6:
        msg = "RANSAC не смог найти устойчивую гомографию."
        return AlignmentResult(False, test, np.ones(ref_gray.shape, dtype=bool), 0, len(matches), msg)

    warped = transform.warp(
        test,
        model.inverse,
        output_shape=reference.shape,
        order=1,
        mode="edge",
        cval=0.0,
        preserve_range=True,
    ).astype(np.float32)

    coverage = transform.warp(
        np.ones(test_gray.shape, dtype=np.float32),
        model.inverse,
        output_shape=ref_gray.shape,
        order=0,
        mode="constant",
        cval=0.0,
        preserve_range=True,
    )
    valid_mask = coverage > 0.5

    return AlignmentResult(True, warped, valid_mask, int(inliers.sum()), int(len(matches)))


@dataclasses.dataclass
class DetectionConfig:
    blur_sigma: float = 1.5
    diff_threshold: float = 3.0  # в сигмах
    min_component_area: int = 120
    closing_size: int = 5
    max_keypoints: int = 2000
    pcb_padding: int = 5
    reference_corners: List[Tuple[float, float]] | None = None
    test_corners: List[Tuple[float, float]] | None = None


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


def align_using_corner_annotations(
    reference: Array,
    test: Array,
    reference_corners: List[Tuple[float, float]],
    test_corners: List[Tuple[float, float]],
    padding: int,
) -> Tuple[Array, Array, Array]:
    """
    Выравнивает плату теста относительно эталона по заранее известным углам.
    Возвращает: (вырезанная эталонная область, трансформированный тест, валидная маска).
    """
    reference_region, ref_bbox, ref_mask = extract_pcb_region(reference, reference_corners, padding=padding)
    test_region, test_bbox, test_mask = extract_pcb_region(test, test_corners, padding=padding)

    H = find_equals(np.asarray(test_corners, dtype=np.float32), np.asarray(reference_corners, dtype=np.float32))
    H_adapted = adapt_homography_for_cropped_regions(H, ref_bbox, test_bbox)

    transformed_test = transform_pcb_to_reference(test_region, H_adapted, reference_region.shape)
    transformed_test_mask = transform_pcb_to_reference(
        (test_mask.astype(np.uint8) * 255),
        H_adapted,
        reference_region.shape[:2],
    ) > 0

    valid_mask = ref_mask & transformed_test_mask

    return reference_region.astype(np.float32), transformed_test.astype(np.float32), valid_mask


def detect_components(reference_path: str, test_path: str, config: DetectionConfig) -> None:
    reference_full = load_image(reference_path)
    test_full = load_image(test_path)

    use_corners = bool(config.reference_corners and config.test_corners)
    reference_for_diff: Array = reference_full
    aligned_test: Array | None = None
    valid_mask: Array | None = None

    if use_corners:
        try:
            print("[i] Используем заданные углы платы для вычисления гомографии.")
            reference_for_diff, aligned_test, valid_mask = align_using_corner_annotations(
                reference_full,
                test_full,
                config.reference_corners or [],
                config.test_corners or [],
                padding=config.pcb_padding,
            )
            print("[i] Гомография по углам успешно применена.")
        except ValueError as err:
            print(f"[!] Не удалось применить гомографию по углам: {err}")
            aligned_test = None

    if aligned_test is None:
        alignment = align_images(reference_full, test_full, max_keypoints=config.max_keypoints)
        if not alignment.success:
            print(f"[!] Автовыравнивание не удалось: {alignment.message}")
            aligned_test = test_full
            valid_mask = np.ones(reference_full.shape[:2], dtype=bool)
        else:
            print(
                f"[i] Выравнивание успешно: inliers {alignment.inliers}/{alignment.total_matches}"
            )
            aligned_test = alignment.warped
            valid_mask = alignment.valid_mask
        reference_for_diff = reference_full

    diff, extra_mask, missing_mask = compute_difference_maps(
        reference_for_diff,
        aligned_test,
        config,
        valid_mask=valid_mask,
    )
    extra_regions = extract_regions(extra_mask, diff, config.min_component_area)
    missing_regions = extract_regions(missing_mask, diff, config.min_component_area)

    print("=== Детектор компонентов ===")
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
    )
    detect_components("chess.jpg", "chessboard_3_3.jpg", config)


if __name__ == "__main__":
    main()
