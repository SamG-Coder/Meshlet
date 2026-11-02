# Meshlet terrain

A terrain-only Three.js **0.187.0 / WebGPU** experiment: streamed meshlets, distance-based HLOD, baked diffuse path-traced lighting, volumetric clouds, falling rain, wet materials and GPU surface-water transport.

## Run

Requires Node.js **22.12 or newer**, a WebGPU-capable browser, and the system `tar` command (included with current Windows, macOS and common Linux distributions).

```sh
git clone https://github.com/SamG-Coder/Meshlet.git
cd Meshlet
npm ci
npm run assets
npm run dev
```

Open **http://127.0.0.1:5173/**. The asset command downloads the versioned release pack, checks its SHA-256, extracts it, and verifies every file. Generated bakes are distributed separately from Git history. Allow roughly 3 GB of disk space for the download and extracted data, plus dependencies/build output.

The current maximum-detail scene was developed on an RTX 5080 and typically allocates roughly 6.4 GiB of tracked GPU resources. Smaller GPUs can use a lower HLOD budget. Hardware WebGPU is required; there is no WebGL fallback. The dependency patch is applied automatically by `npm ci`.

## Explore

- Click or drag to look; WASD moves. F toggles walking/flying. Q/E moves vertically in fly mode; Shift increases speed.
- Maximum HLOD and full resident detail are the defaults. Region, Ridge, Ground and River select inspection views.
- Lighting / clouds controls cloud cover, exposure and cloud animation.
- Rain / rivers controls rainfall and accelerated hydrology. Rain falls at real-time speed independently of the hydrology clock. Rain needs cloud coverage above the area.
- Stop rain retains accumulated water and moisture. Reset water clears the simulation.
- Measure settled frames and Test HLOD movement record actual frame intervals and coverage errors.

## What is implemented

The map spans **32.768 km**, with an **8,192² source heightfield** (about 4 m spacing). Four workers prepare terrain tiles. HLOD morphing and balanced neighbours reduce chunk changes; reusable GPU pools limit streaming churn. The finest render grid interpolates the source and adds shallow material relief. The displayed **549.76 billion potential triangles** are a theoretical complete fine grid, not resident geometry or simulation cells.

The lighting release contains 4,096² sky and sun-bounce irradiance maps, traced offline with 256 samples and up to six diffuse bounces, plus 8,192² sun visibility. Moving clouds modulate the light and rainfall. Ten generated material layers use stochastic triplanar sampling.

Water uses a GPU virtual-pipe heightfield model with persistent flux, rainfall, infiltration, evaporation and drainage. Regional cells are 32 m; a camera-following 256 m patch uses 1 m cells. Thin water remains a wet material film; deeper water uses reconstructed free-surface geometry and depth-dependent transparency. Near and middle rain particles overlap volumetric rain shafts. UV clamping and centroid interpolation prevent invalid highlights on subpixel rain triangles.

## Verification

```sh
npm test
npm run assets:check
npm run build
```

For actual GPU hydraulic fixtures, open `/water-test.html` or press **Check flow / water balance** in the terrain. These check dry start, downhill transport into non-raining cells, water retention after rain, recession, a still lake and a dry-day fixture. An observed result is included in `verification/hydraulic-verification.json`.

## Limits

This is an experimental graphics simulation, not a calibrated flood or geological forecast. The retained landscape has artificial drainage patterns. No scientifically validated 50,000-year erosion claim is made. The regional and local water grids are not a fully coupled, region-wide 1 m fluid solver. Recentring preserves overlapping local storage but resets its momentum. Water optics, cloud multiple scattering and dynamic indirect-light changes are approximations. The baked lighting traces the source heightfield, not every centimetre of material relief.

See [baking instructions](docs/BAKING.md), [asset provenance](THIRD_PARTY_NOTICES.md), and [ISC license](LICENSE). This repository intentionally contains only terrain-related code and data.
