import argparse
from pathlib import Path
import geopandas as gpd
import geoai

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--aoi", required=True)
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--dataset", default="google")
args = parser.parse_args()

AOI = f"data/raw/boundaries/{args.aoi}.shp"
YEAR = args.year
DATASET = args.dataset

OUT = Path("data/raw/embeddings") / args.aoi / str(YEAR)
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# LOAD AOI
# -----------------------------------------
print("Loading AOI...")
aoi = gpd.read_file(AOI).to_crs("EPSG:4326")

minx, miny, maxx, maxy = aoi.total_bounds

bbox = [minx, miny, maxx, maxy]

print("AOI bbox:", bbox)

# -----------------------------------------
# SELECT DATASET
# -----------------------------------------
if DATASET == "google":

    dataset = "google/embeddings"

elif DATASET == "tessera":

    dataset = "tessera/embeddings"

else:

    raise ValueError("Dataset must be 'google' or 'tessera'")

print("Dataset:", dataset)

# -----------------------------------------
# DOWNLOAD EMBEDDINGS
# -----------------------------------------
print("\n⬇ Downloading embeddings...")

geoai.download(
    dataset=dataset,
    bbox=bbox,
    year=YEAR,
    output=str(OUT)
)

print("\n✅ Embedding download complete")
print("Saved to:", OUT)