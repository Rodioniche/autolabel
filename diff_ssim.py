from PIL import Image, ImageDraw
import numpy as np

name_1 = "9_5_2"
name_2 = "9_5"

image1_path = 'dst_' + name_1 + '_lab.png'
image2_path = 'dst_' + name_2 + '_lab.png'
diff_name = "results\\diff_" + name_1 + "_from_" + name_2 + ".jpg"
defect_name = "results\\defect_" + name_1 + "_from_" + name_2 + ".jpg"
def compute_ssim(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    covariance = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * covariance + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

    return numerator / denominator if denominator != 0 else 1.0

def compare_images_with_ssim_pil(ref_path, test_path, block_size=16, diff_thresh=30, ssim_thresh=0.75):
    # Загружаем изображения
    ref_img = Image.open(ref_path).convert("L")
    test_img_gray = Image.open(test_path).convert("L")
    test_img_color = Image.open(ref_path).convert("RGB")

    ref_array = np.array(ref_img)
    test_array = np.array(test_img_gray)

    result_img = test_img_color.copy()
    draw = ImageDraw.Draw(result_img)

    defects_img = Image.new("RGB", result_img.size, color=(255, 255, 255))
    defects_pixels = defects_img.load()
    test_pixels_color = test_img_color.load()

    height, width = ref_array.shape

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block_ref = ref_array[y:y+block_size, x:x+block_size]
            block_test = test_array[y:y+block_size, x:x+block_size]

            if block_ref.shape != (block_size, block_size) or block_test.shape != (block_size, block_size):
                continue

            diff_pixels = np.abs(block_ref - block_test)
            num_diffs = np.sum(diff_pixels > diff_thresh)

            if num_diffs >  (block_size * block_size) // 4:
                ssim_score = compute_ssim(block_ref, block_test)
                if ssim_score < ssim_thresh:
                    # Обводим на оригинале
                    draw.rectangle([x, y, x+block_size-1, y+block_size-1], outline=(255, 0, 0))

                    # Копируем цветной блок на белый фон
                    for j in range(block_size):
                        for i in range(block_size):
                            if (y + j < height) and (x + i < width):
                                defects_pixels[x + i, y + j] = test_pixels_color[x + i, y + j]

    return result_img, defects_img

# Пример использования
vis_img, defects_only = compare_images_with_ssim_pil(image1_path, image2_path)
vis_img.show()
vis_img.save(diff_name)

defects_only.show()
defects_only.save(defect_name)