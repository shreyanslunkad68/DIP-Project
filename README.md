# Automated Land-Cover Classification via Spectral Band Ratioing and Spatial Texture Analysis for Defense GIS

A practical Python prototype based on the supplied DIP poster. It implements:
- Input satellite raster loading (GeoTIFF recommended)
- Image enhancement using CLAHE
- Histogram comparison
- Spectral ratio feature when suitable bands are available (e.g., NIR / Red)
- Spatial texture features using local variance
- K-means segmentation (`K=5` by default)
- Canny edge detection
- Export of visualization images and a classified GeoTIFF when the input is georeferenced

## Important scope note
The poster describes spectral band ratioing and spatial texture analysis, but its results table explicitly lists CLAHE, histogram analysis, K-means, Canny, and QGIS. This prototype adds a simple NIR/Red ratio and local-variance texture feature so the implementation better matches the title. It is a research/educational prototype, not a validated operational defense system. Clusters are not automatically semantic labels such as forest, water, road, or urban.

## 1. Setup (Windows PowerShell)

Use Python 3.10 or 3.11 for the smoothest compatibility with geospatial/scientific libraries.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already use a compatible Python environment, activate it and install the requirements directly.

## 2. Input data

Best input: a multispectral GeoTIFF, such as a Sentinel-2 tile, with known band order. The script assumes:
- 3-band input: bands 1, 2, 3 are treated as RGB for display; no NIR ratio is computed.
- 4-or-more-band input: bands 1, 2, 3 are displayed as RGB and band 4 is treated as NIR by default for the NIR/Red ratio. **Check your file's actual band order before using it.** Sentinel-2 products often use different band resolutions and ordering depending on how they were exported.

You can also use a regular PNG/JPG for a visual demonstration. In that case, the spectral ratio is skipped because RGB imagery does not contain a true NIR band.

## 3. Run

```powershell
python main.py --input data/satellite.tif --output outputs
```

For an RGB image:

```powershell
python main.py --input data/satellite.jpg --output outputs
```

Optional arguments:
```powershell
python main.py --input data/satellite.tif --output outputs --clusters 5 --nir-band 4 --red-band 1
```

Band indices are **1-based** in command-line arguments.

## 4. Outputs

The `outputs/` folder contains:
- `original.png`
- `enhanced.png`
- `histogram.png`
- `kmeans_clusters.png`
- `edges.png`
- `texture.png`
- `spectral_ratio.png` (only when a suitable NIR band exists)
- `classified.tif` (when input is a georeferenced GeoTIFF)
- `run_summary.json`

The cluster IDs are numbered 0 to K-1 and have no inherent land-cover meaning. For semantic land-cover labels and accuracy metrics, provide ground-truth labels and train/evaluate a supervised model.

## 5. What to explain in the poster presentation

1. Preprocessing and CLAHE improve visual contrast.
2. Histogram comparison visualizes intensity distribution changes.
3. The NIR/Red ratio is a simple spectral feature when real multispectral bands are available.
4. Local variance is a simple texture descriptor that captures neighborhood variation.
5. K-means groups pixels using selected features; it does not itself know terrain classes.
6. Canny highlights image boundaries.
7. QGIS can display the classified GeoTIFF if it retains valid georeferencing.

## Limitations
- The default band mapping may not match every satellite product; inspect metadata and change band arguments.
- Local variance is a basic texture descriptor, not a complete texture model.
- K-means is sensitive to feature scaling and the chosen number of clusters.
- Quantitative accuracy requires reference labels/ground truth.
- Georeferenced export requires a georeferenced input raster.
