"""Render the aerofoil solution to frames. pvbatch --force-offscreen-rendering render_aerofoil.py [n]

Colour is velocity magnitude on the cell data, on a single-hue ramp: this is a magnitude field, so a
sequential ramp is the right encoding and a rainbow would invent structure that is not in the data.
"""
import sys
from paraview.simple import *

CASE = "/home/ghost/space/visual_aerofoil_64k/case.foam"
OUT  = "/home/ghost/space/aerofoil64k_frames"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0        # 0 = every frame

import os
os.makedirs(OUT, exist_ok=True)

r = OpenFOAMReader(registrationName="case", FileName=CASE)
r.MeshRegions = ["internalMesh"]
r.CellArrays  = ["U", "p"]
r.UpdatePipeline()

times = list(r.TimestepValues or [])
print("timesteps:", len(times), times[:3], "...", times[-3:] if times else "")

b = r.GetDataInformation().GetBounds()
span = [b[1]-b[0], b[3]-b[2], b[5]-b[4]]
thin = span.index(min(span))                                 # the extruded direction
print("bounds", b, "-> thin axis", thin)

v = GetActiveViewOrCreate("RenderView")
v.ViewSize = [1920, 1080]
v.OrientationAxesVisibility = 0
v.Background = [0.043, 0.055, 0.094]                         # deep indigo, matches the promo ground
v.UseColorPaletteForBackground = 0

d = Show(r, v)
d.Representation = "Surface"
ColorBy(d, ("CELLS", "U", "Magnitude"))
lut = GetColorTransferFunction("U")
lut.ApplyPreset("Viridis (matplotlib)", True)                # single perceptual ramp, not a rainbow
d.SetScalarBarVisibility(v, False)

# look straight down the thin axis, tight on the aerofoil rather than the whole far field
cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
n = [0, 0, 0]; n[thin] = 1
up = [0, 0, 1] if thin != 2 else [0, 1, 0]
# view from the far side of the thin axis: the default put the leading edge on the right, so the
# flow read right-to-left. Mirroring the camera puts the freestream where a reader expects it.
v.CameraPosition = [cx - n[0]*max(span)*2, cy - n[1]*max(span)*2, cz - n[2]*max(span)*2]
v.CameraFocalPoint = [cx, cy, cz]
v.CameraViewUp = up
ResetCamera()
v.CameraParallelProjection = 1
v.CameraParallelScale = v.CameraParallelScale * 0.34          # zoom to the aerofoil

sel = times if not LIMIT else times[::max(1, len(times)//LIMIT)][:LIMIT]
for i, t in enumerate(sel):
    v.ViewTime = t
    r.UpdatePipeline(t)
    if i == 0:
        d.RescaleTransferFunctionToDataRange(False, True)
        lut.RescaleTransferFunction(0.0, 340.0)               # fixed range: colour must not shift per frame
    SaveScreenshot(f"{OUT}/f{i:04d}.png", v, ImageResolution=[1920, 1080])
    if i % 25 == 0:
        print("frame", i, "t", t, flush=True)
print("done", len(sel), "frames ->", OUT)
