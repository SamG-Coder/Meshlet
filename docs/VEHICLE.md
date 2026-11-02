# Optional Raptor vehicle

The terrain supports a locally supplied **2024 Ford F-150 Raptor R** GLB. The model is not included in Git or the terrain asset release.

Download the original GLB ZIP from [the model page](https://sketchfab.com/3d-models/2024-ford-f-150-raptor-r-91aedf1119724f3396dcfe87686545c1), then run:

```sh
python scripts/install-vehicle.py /path/to/2024-ford-f-150-raptor-r.zip
```

This extracts only `source/2024_ford_f-150_raptor_r.glb` into the ignored `public/assets/vehicles/raptor/` directory. Embedded textures are already inside the original GLB. Reload the terrain page after installation. Without the asset, terrain exploration remains available and vehicle controls remain disabled.

## Controls

- W: accelerate; S: brake, then reverse.
- A/D: steer, with speed-sensitive steering and Ackermann front-wheel angles.
- Space: rear-wheel handbrake.
- Hold the left mouse button and drag: orbit the chase camera.
- V: enter/exit driving mode.
- Flip upright: right the vehicle at its current position, preserving heading.
- R / Recover: relocate to a flatter nearby footprint.
- Return to valley start: reset to the initial valley parking area.

## Implementation

Rapier runs a 120 Hz fixed-step rigid-body simulation with four raycast spring/damper contacts. The imported tyres/rims rotate independently; brake calipers follow suspension and steering without spinning. Collision uses six body pieces plus four tyre cylinders, with continuous collision detection. This is a game-tuned approximation; suspension links are not a fully articulated mechanical model.

The collision terrain is sampled from the full source heightfield plus material relief, independently of visible HLOD. Persistent 32 m collision tiles have 0.5 m vertex spacing and preload around the vehicle. Tiles beneath the vehicle are retained when render chunks change. The underlying source heightfield still has approximately 4 m spacing: 0.5 m collision sampling does not invent additional erosion data.

The vehicle uses a separate standard geometry pass alongside the meshlet terrain. Materials receive local baked irradiance, sky/cloud reflections and dynamic sun lighting. A local terrain receiver catches a live vehicle shadow. The moving vehicle is not part of the offline terrain path trace.

## Validation and limits

`npm test` includes deterministic checks for static support, acceleration, braking, steering, reverse and uneven-ground suspension. A source-map trial also traversed approximately 122 m across streamed collision tiles with four tyre contacts at the end. These checks do not establish real Ford handling accuracy.

The current work includes the requested mouse orbit and recovery controls. Terrain white-strip/edge artifacts reported during driving are still under investigation; this commit does not claim that rendering defect is resolved.

References: [Rapier vehicle controller](https://rapier.rs/docs/user_guides/javascript/vehicle_controller/), [Ford 2024 Raptor specifications](https://media.ford.com/content/dam/fordmedia/North%20America/US/product/2024/f150/2024%20Ford%20Raptor%20and%20Raptor%20R%20Tech%20Specs.pdf).

## Asset attribution

The supplied GLB identifies the title as “2024 Ford F-150 Raptor R”, author as **Ddiaz Design** (`https://sketchfab.com/ddiaz-design`), and license as **CC BY-NC-SA 4.0**. The user's download-page screenshot additionally credits **VTX**. Preserve the original metadata and verify the current model-page attribution before redistributing the asset. The project's ISC license does not apply to this third-party vehicle. No vehicle asset is redistributed by this repository.
