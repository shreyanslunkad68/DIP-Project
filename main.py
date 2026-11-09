import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from skimage.filters.rank import entropy
from skimage.morphology import disk


def normalize_band(band):
    """Scale a numeric band robustly to uint8 for visualization/features."""
    band = np.asarray(band, dtype=np.float32)
    valid = np.isfinite(band)
    if not valid.any():
        return np.zeros(band.shape, dtype=np.uint8)
    lo, hi = np.percentile(band[valid], [2, 98])
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)
    scaled = np.clip((band - lo) / (hi - lo), 0, 1)
    scaled[~valid] = 0
    return (scaled * 255).astype(np.uint8)


def read_input(path, red_band=1, nir_band=4, max_side=1400):
    """Read raster. Band arguments are 1-based. Returns display RGB and metadata."""
    path = str(path)
    if path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(path) as src:
            count = src.count
            if count < 1:
                raise ValueError("Raster contains no bands.")
            scale = min(1.0, max_side / max(src.height, src.width))
            out_h = max(1, int(src.height * scale))
            out_w = max(1, int(src.width * scale))
            indexes = list(range(1, min(3, count) + 1))
            rgb_bands = []
            for idx in indexes:
                arr = src.read(idx, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
                rgb_bands.append(normalize_band(arr))
            if len(rgb_bands) == 1:
                rgb = np.repeat(rgb_bands[0][:, :, None], 3, axis=2)
            elif len(rgb_bands) == 2:
                rgb = np.stack([rgb_bands[0], rgb_bands[1], rgb_bands[1]], axis=2)
            else:
                rgb = np.stack(rgb_bands[:3], axis=2)

            red = None
            nir = None
            if 1 <= red_band <= count:
                red = src.read(red_band, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
            if 1 <= nir_band <= count and count >= 4:
                nir = src.read(nir_band, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
            profile = src.profile.copy()
            # Keep georeferencing while updating dimensions and writing a one-band output.
            profile.update(height=out_h, width=out_w, count=1, dtype="uint8", nodata=0, compress="lzw")
            return rgb, red, nir, profile, True, count

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read input image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, None, None, None, False, 3


def make_features(rgb, red, nir):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Local variance approximates neighborhood texture.
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (9, 9))
    mean_sq = cv2.blur(gray_f * gray_f, (9, 9))
    local_var = np.maximum(mean_sq - mean * mean, 0)
    texture_u8 = normalize_band(local_var)

    spectral_ratio = None
    if red is not None and nir is not None:
        denom = nir + red
        # Use normalized difference ratio (NIR - Red) / (NIR + Red).
        # This is an NDVI-style feature; it is meaningful only for correctly mapped NIR and Red bands.
        ratio = np.divide(nir - red, denom, out=np.zeros_like(nir), where=np.abs(denom) > 1e-8)
        spectral_ratio = np.clip(ratio, -1, 1).astype(np.float32)

    feature_list = [
        rgb.astype(np.float32) / 255.0,
        (enhanced.astype(np.float32) / 255.0)[:, :, None],
        (texture_u8.astype(np.float32) / 255.0)[:, :, None],
    ]
    if spectral_ratio is not None:
        feature_list.append(spectral_ratio[:, :, None])
    features = np.concatenate(feature_list, axis=2)
    return gray, enhanced, texture_u8, spectral_ratio, features


def save_rgb(path, image):
    plt.figure(figsize=(10, 7))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_gray(path, image, title=None, cmap="gray"):
    plt.figure(figsize=(10, 7))
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_histogram(path, original_gray, enhanced_gray):
    plt.figure(figsize=(9, 5))
    plt.hist(original_gray.ravel(), bins=256, range=(0, 255), alpha=0.55, label="Original")
    plt.hist(enhanced_gray.ravel(), bins=256, range=(0, 255), alpha=0.55, label="CLAHE enhanced")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Pixel count")
    plt.title("Histogram comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="DIP + GIS land-cover segmentation prototype")
    parser.add_argument("--input", required=True, help="Input satellite image (.tif/.tiff/.png/.jpg)")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--clusters", type=int, default=5, help="Number of K-means clusters")
    parser.add_argument("--red-band", type=int, default=1, help="1-based red band index for multispectral GeoTIFF")
    parser.add_argument("--nir-band", type=int, default=4, help="1-based NIR band index; ratio skipped if unavailable")
    args = parser.parse_args()

    if args.clusters < 2:
        raise ValueError("--clusters must be at least 2")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rgb, red, nir, profile, is_geotiff, band_count = read_input(
        args.input, args.red_band, args.nir_band
    )
    gray, enhanced, texture, spectral_ratio, features = make_features(rgb, red, nir)

    h, w = gray.shape
    X = features.reshape(-1, features.shape[2])
    # Standardize features so one feature's numeric range does not dominate.
    X = StandardScaler().fit_transform(X)

    model = MiniBatchKMeans(
        n_clusters=args.clusters,
        random_state=42,
        batch_size=4096,
        n_init=5
    )
    labels = model.fit_predict(X).reshape(h, w).astype(np.uint8)

    # Give clusters display colours. Cluster IDs remain arbitrary.
    rng = np.random.default_rng(42)
    palette = rng.integers(35, 245, size=(args.clusters, 3), dtype=np.uint8)
    clustered_rgb = palette[labels]

    edges = cv2.Canny(enhanced, 80, 160)

    save_rgb(out / "original.png", rgb)
    save_gray(out / "enhanced.png", enhanced, "CLAHE enhanced image")
    save_histogram(out / "histogram.png", gray, enhanced)
    save_rgb(out / "kmeans_clusters.png", clustered_rgb)
    save_gray(out / "edges.png", edges, "Canny edge detection")
    save_gray(out / "texture.png", texture, "Local variance texture feature")

    if spectral_ratio is not None:
        plt.figure(figsize=(10, 7))
        plt.imshow(spectral_ratio, cmap="RdYlGn", vmin=-1, vmax=1)
        plt.colorbar(label="(NIR - Red) / (NIR + Red)")
        plt.title("Normalized difference spectral feature")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out / "spectral_ratio.png", dpi=160, bbox_inches="tight")
        plt.close()

    if is_geotiff and profile is not None:
        with rasterio.open(out / "classified.tif", "w", **profile) as dst:
            dst.write(labels, 1)

    summary = {
        "input": os.path.abspath(args.input),
        "height": int(h),
        "width": int(w),
        "input_band_count": int(band_count),
        "clusters": int(args.clusters),
        "features_used": ["RGB", "CLAHE intensity", "local variance texture"] +
                         (["normalized difference (NIR-Red)/(NIR+Red)"] if spectral_ratio is not None else []),
        "spectral_ratio_computed": spectral_ratio is not None,
        "georeferenced_geotiff_exported": bool(is_geotiff),
        "warning": "Cluster IDs are unsupervised groups, not verified semantic land-cover classes. Accuracy requires ground truth."
    }
    with open(out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Outputs saved to: {out.resolve()}")
    print("Clusters are not semantic labels. Validate against reference data before reporting accuracy.")
    if spectral_ratio is None:
        print("No NIR/Red ratio computed. Use a suitable multispectral GeoTIFF and verify band mapping.")


if __name__ == "__main__":
    main()
