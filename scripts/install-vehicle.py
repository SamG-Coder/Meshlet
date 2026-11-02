"""Install the user's original Raptor GLB locally, without publishing it."""
from pathlib import Path
import argparse
import shutil
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("zip", type=Path, help="Original 2024-ford-f-150-raptor-r.zip download")
args = parser.parse_args()
member = "source/2024_ford_f-150_raptor_r.glb"
target = Path(__file__).resolve().parents[1] / "public/assets/vehicles/raptor" / member
with zipfile.ZipFile(args.zip) as archive:
    info = archive.getinfo(member)
    if info.file_size > 512 * 1024 * 1024:
        raise ValueError("Unexpected model size")
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, target.open("wb") as output:
        shutil.copyfileobj(source, output)
print(f"Installed {target}")
print("Third-party CC BY-NC-SA asset; see docs/VEHICLE.md for attribution.")
