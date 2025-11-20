import dataclasses
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
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


def detect_components(reference_path: str, test_path: str, config: DetectionConfig) -> None:
    reference = load_image(reference_path)
    test = load_image(test_path)

    alignment = align_images(reference, test, max_keypoints=config.max_keypoints)
    if not alignment.success:
        print(f"[!] Автовыравнивание не удалось: {alignment.message}")
        aligned_test = test
        valid_mask = np.ones(reference.shape[:2], dtype=bool)
    else:
        print(
            f"[i] Выравнивание успешно: inliers {alignment.inliers}/{alignment.total_matches}"
        )
        aligned_test = alignment.warped
        valid_mask = alignment.valid_mask

    diff, extra_mask, missing_mask = compute_difference_maps(
        reference,
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

    visualize(reference, aligned_test, diff, extra_regions, missing_regions)


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
