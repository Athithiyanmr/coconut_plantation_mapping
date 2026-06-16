"""
calculate_area.py
-----------------
Compute coconut plantation area from the binary prediction GeoTIFF
produced by scripts/dl/predict_unet.py.

Output files:
    outputs/unet/<year>/coconut_area_<year>_<aoi>.txt   -- plain text summary
    outputs/unet/<year>/coconut_area_<year>_<aoi>.csv   -- CSV for logging

Usage (standalone):
    python scripts/calculate_area.py --year 2026 --aoi villupuram --threshold 0.35

Or pass the binary tif directly:
    python scripts/calculate_area.py --pred_tif outputs/unet/2026/coconut_binary_2026_villupuram.tif

Auto-integrated at end of run.py pipeline.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ── Conversion constants ─────────────────────────────────────────────────────
M2_PER_HA    = 10_000.0
ACRES_PER_HA = 2.47105


def _reproject_to_utm(src_path: Path) -> tuple:
    """Return (data_array, pixel_area_m2) after reprojecting to UTM if needed."""
    with rasterio.open(src_path) as src:
        crs = src.crs
        if crs and crs.is_projected:
            # Already metric — read directly
            data        = src.read(1)
            pixel_w     = abs(src.transform.a)
            pixel_h     = abs(src.transform.e)
            pixel_area  = pixel_w * pixel_h
            return data, pixel_area, crs

        # Geographic CRS (degrees) — reproject to UTM for accurate area
        print("  CRS is geographic — reprojecting to UTM for area calculation...")
        dst_crs = CRS.from_epsg(32644)  # UTM Zone 44N  (Tamil Nadu / AP)
        transform_utm, width_utm, height_utm = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            crs=dst_crs,
            transform=transform_utm,
            width=width_utm,
            height=height_utm,
            nodata=255,
        )
        reprojected = np.zeros((height_utm, width_utm), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform_utm,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        pixel_w    = abs(transform_utm.a)
        pixel_h    = abs(transform_utm.e)
        pixel_area = pixel_w * pixel_h
        return reprojected, pixel_area, dst_crs


def calculate_area(pred_tif: Path, year: str, aoi: str) -> dict:
    """
    Read binary prediction raster (1=coconut, 0=background, 255=nodata).
    Returns a dict with pixel count, m2, hectares, and acres.
    """
    pred_tif = Path(pred_tif)
    if not pred_tif.exists():
        sys.exit(f"\n[ERROR] File not found: {pred_tif}\n"
                 f"  Make sure prediction has been run for {aoi} {year}.")

    print(f"\n{'='*58}")
    print(f"  COCONUT AREA CALCULATION")
    print(f"  AOI      : {aoi.upper()}   |   Year : {year}")
    print(f"  File     : {pred_tif}")
    print(f"{'='*58}")

    data, pixel_area_m2, final_crs = _reproject_to_utm(pred_tif)

    # Count only class=1 pixels (exclude 0=background and 255=nodata)
    coconut_pixels = int(np.sum(data == 1))
    valid_pixels   = int(np.sum(data != 255))  # AOI pixels
    total_pixels   = data.size

    total_m2    = coconut_pixels * pixel_area_m2
    total_ha    = total_m2   / M2_PER_HA
    total_acres = total_ha   * ACRES_PER_HA
    pct_aoi     = 100.0 * coconut_pixels / valid_pixels if valid_pixels > 0 else 0.0

    print(f"\n  CRS              : {final_crs.to_string()}")
    print(f"  Pixel size       : {pixel_area_m2**0.5:.1f} m × {pixel_area_m2**0.5:.1f} m")
    print(f"  Pixel area       : {pixel_area_m2:.2f} m²")
    print(f"\n  Coconut pixels   : {coconut_pixels:>12,}")
    print(f"  Valid AOI pixels : {valid_pixels:>12,}")
    print(f"  Coconut coverage : {pct_aoi:>11.2f} % of AOI")
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Area = {total_m2:>16,.2f}  m²      │")
    print(f"  │  Area = {total_ha:>16,.4f}  hectares │")
    print(f"  │  Area = {total_acres:>16,.4f}  acres    │")
    print(f"  └─────────────────────────────────────┘")
    print(f"{'='*58}\n")

    return {
        "year":            year,
        "aoi":             aoi,
        "coconut_pixels":  coconut_pixels,
        "valid_pixels":    valid_pixels,
        "pixel_area_m2":   pixel_area_m2,
        "area_m2":         total_m2,
        "area_ha":         total_ha,
        "area_acres":      total_acres,
        "pct_aoi":         pct_aoi,
        "pred_tif":        str(pred_tif),
    }


def save_results(result: dict, out_dir: Path):
    """Write plain-text summary and CSV to outputs/unet/<year>/."""
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"coconut_area_{result['year']}_{result['aoi']}.txt"
    csv_path = out_dir / f"coconut_area_{result['year']}_{result['aoi']}.csv"

    # ── TXT ──────────────────────────────────────────────────────────────────
    lines = [
        "COCONUT PLANTATION AREA REPORT",
        "=" * 42,
        f"AOI             : {result['aoi'].upper()}",
        f"Year            : {result['year']}",
        f"Prediction file : {result['pred_tif']}",
        "",
        f"Pixel area      : {result['pixel_area_m2']:.2f} m²",
        f"Coconut pixels  : {result['coconut_pixels']:,}",
        f"Valid AOI pixels: {result['valid_pixels']:,}",
        f"Coverage        : {result['pct_aoi']:.2f} % of AOI",
        "",
        f"Area (m²)       : {result['area_m2']:,.2f}",
        f"Area (hectares) : {result['area_ha']:,.4f}",
        f"Area (acres)    : {result['area_acres']:,.4f}",
        "=" * 42,
    ]
    txt_path.write_text("\n".join(lines))
    print(f"  Saved: {txt_path}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    fieldnames = [
        "year", "aoi", "coconut_pixels", "valid_pixels",
        "pixel_area_m2", "area_m2", "area_ha", "area_acres",
        "pct_aoi", "pred_tif",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(result)
    print(f"  Saved: {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate coconut plantation area from binary prediction raster"
    )
    # Option A: auto-resolve path from year + aoi
    parser.add_argument("--year",      default=None, help="Year used in pipeline (e.g. 2026)")
    parser.add_argument("--aoi",       default=None, help="District name (e.g. villupuram)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold used (only needed if non-standard filename)")

    # Option B: pass binary tif directly
    parser.add_argument("--pred_tif",  default=None,
                        help="Direct path to binary .tif (overrides --year/--aoi)")

    args = parser.parse_args()

    # ── Resolve file path ─────────────────────────────────────────────────────
    if args.pred_tif:
        pred_path = Path(args.pred_tif)
        # Try to parse year/aoi from filename if not supplied
        if args.year is None or args.aoi is None:
            stem_parts = pred_path.stem.split("_")
            # Expected stem: coconut_binary_<year>_<aoi>
            try:
                year = stem_parts[2]
                aoi  = "_".join(stem_parts[3:])
            except IndexError:
                year = "unknown"
                aoi  = "unknown"
        else:
            year = args.year
            aoi  = args.aoi
    else:
        if not args.year or not args.aoi:
            parser.error("Provide either --pred_tif OR both --year and --aoi")
        year      = args.year
        aoi       = args.aoi
        pred_path = Path(f"outputs/unet/{year}/coconut_binary_{year}_{aoi}.tif")

    out_dir = Path(f"outputs/unet/{year}")
    result  = calculate_area(pred_path, year, aoi)
    save_results(result, out_dir)
