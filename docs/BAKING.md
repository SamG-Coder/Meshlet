# Terrain bakes

The release pack is the reproducible runtime snapshot. Python is optional for viewing it. The included offline scripts use NumPy, Pillow, SciPy, Numba and PyTorch; CUDA is required for terrain generation and light tracing.

```sh
python scripts/bake-generated-ground.py
python scripts/bake-cloud-noise.py
python scripts/bake-rain-impacts.py
python scripts/bake-terrain-hydrology.py
python scripts/bake-terrain-lighting.py
python scripts/verify-baked-lighting.py
```

The light baker currently targets Windows, PyTorch's CUDA 12 runtime/NVRTC and compute capability 12.0 (RTX 5080). Port its DLL discovery and compiler target before using a different platform/GPU. It writes linear, little-endian RGBA16F irradiance data; alpha is unused. The sunlight visibility image is R8. Binary maps use row zero at negative world Z.

The original height generation stages are included for investigation:

```sh
python scripts/erode-terrain.py --size 8192 --steps 1600 --out public/assets/erosion-v3
python scripts/build-badlands.py --source public/assets/erosion-v3 --out public/assets/badlands-v3
```

These are memory-intensive artistic terrain experiments. Rebuilding can differ across numerical environments and requires rebaking lighting/hydrology. Use the hashed release snapshot to reproduce the published viewer. Old tree work, alternate rejected landscapes and one-off source-rewriting scripts are not part of this repository.

The runtime uses current simulated water depth and flux; the D8 paths output by the hydrology baker are diagnostics and a camera-view selection aid, not rendered river ribbons.

References: [GPU virtual-pipe hydraulic erosion](https://evasion.imag.fr/Publications/2007/MDH07/), [local-inertial flood modelling](https://research-information.bris.ac.uk/en/publications/a-simple-inertial-formulation-of-the-shallow-water-equations-for-/), [PBRT path tracing](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/Path_Tracing). The implementations here are graphics approximations, not validated reproductions of every model in these sources.
