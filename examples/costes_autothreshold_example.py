from pathlib import Path

import tifffile

from coloc_tools import fiji_bisection_auto_threshold, fiji_costes_auto_threshold

image_path = Path(__file__).parent / "test_image.tif"
image = tifffile.imread(image_path)
ch1, ch2 = image[0], image[1]

ch1_thr_bic, ch2_thr_bic, slope_bic, intercept_bic = fiji_bisection_auto_threshold(
    ch1, ch2
)
ch1_thr_cos, ch2_thr_cos, slope_cos, intercept_cos = fiji_costes_auto_threshold(
    ch1, ch2
)

print("\nBisection Method Results:")
print(f"y = {slope_bic:.4f} * x + {intercept_bic:.4f}")
print(f"Ch1 Threshold: {ch1_thr_bic}, Ch2 Threshold: {ch2_thr_bic}")
print("\nCostes Method Results:")
print(f"y = {slope_cos:.4f} * x + {intercept_cos:.4f}")
print(f"Ch1 Threshold: {ch1_thr_cos}, Ch2 Threshold: {ch2_thr_cos}")
