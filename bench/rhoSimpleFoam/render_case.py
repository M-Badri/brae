"""Render an OpenFOAM case to frames. pvbatch render_case.py <caseDir> <outDir> [maxFrames]

Picks the view from the mesh itself: a thin (extruded) mesh is shown face-on as a surface; a genuinely
three-dimensional one is cut through its centre, because the outside of a pipe tells you nothing about
the mixing inside it. Colour is velocity magnitude on a single perceptual ramp -- a magnitude field
wants a sequential ramp, and a rainbow would invent structure the data does not have.
"""
import sys, os
from paraview.simple import *

CASE, OUT = sys.argv[1], sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(OUT, exist_ok=True)
foam = os.path.join(CASE, "case.foam")
open(foam, "a").close()

r = OpenFOAMReader(registrationName="case", FileName=foam)
r.MeshRegions = ["internalMesh"]
r.CellArrays = ["U", "p"]
r.UpdatePipeline()
times = list(r.TimestepValues or [])
b = r.GetDataInformation().GetBounds()
span = [b[1]-b[0], b[3]-b[2], b[5]-b[4]]
thin = span.index(min(span))
flat = min(span) < 0.12 * max(span)          # extruded 2D case?
print("timesteps", len(times), "bounds", b, "flat" if flat else "3D")

src = r
if not flat:                                  # cut through the middle of the smallest extent
    n = [0, 0, 0]; n[thin] = 1
    src = Slice(registrationName="cut", Input=r)
    src.SliceType.Origin = [(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2]
    src.SliceType.Normal = n
    src.UpdatePipeline()

v = GetActiveViewOrCreate("RenderView")
# VIS_SIZE lets a long thin domain be rendered at its own aspect. injectorPipe is 6.5:1; forcing it
# into 16:9 leaves a strip of flow in a field of black, which is what the first render did.
import os as _e
_sz = _e.environ.get("VIS_SIZE", "1920x1080").split("x")
v.ViewSize = [int(_sz[0]), int(_sz[1])]
v.OrientationAxesVisibility = 0
v.UseColorPaletteForBackground = 0
v.Background = [0.043, 0.055, 0.094]
d = Show(src, v); d.Representation = "Surface"
ColorBy(d, ("CELLS", "U", "Magnitude"))
lut = GetColorTransferFunction("U"); lut.ApplyPreset("Viridis (matplotlib)", True)
d.SetScalarBarVisibility(v, False)

cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
nrm = [0, 0, 0]; nrm[thin] = 1
v.CameraPosition = [cx - nrm[0]*max(span)*2, cy - nrm[1]*max(span)*2, cz - nrm[2]*max(span)*2]
v.CameraFocalPoint = [cx, cy, cz]
v.CameraViewUp = [0, 0, 1] if thin != 2 else [0, 1, 0]
ResetCamera(); v.CameraParallelProjection = 1
# A long thin domain (injectorPipe is 6.5:1) fits a 16:9 frame only by shrinking to a band with black
# above and below. Frame on the interesting end instead: VIS_SCALE sets the half-height, VIS_CX/CY/CZ
# shift the focal point, VIS_VMAX pins the colour range so the field is not crushed into the dark end
# by one fast cell.
import os as _os
_sc = _os.environ.get("VIS_SCALE")
v.CameraParallelScale = float(_sc) if _sc else v.CameraParallelScale * 0.55
for _k, _i in (("VIS_CX", 0), ("VIS_CY", 1), ("VIS_CZ", 2)):
    if _os.environ.get(_k):
        v.CameraFocalPoint[_i] = float(_os.environ[_k])
        v.CameraPosition[_i] = float(_os.environ[_k]) + (v.CameraPosition[_i] - v.CameraFocalPoint[_i])

sel = times if not LIMIT else times[::max(1, len(times)//LIMIT)][:LIMIT]
for i, t in enumerate(sel):
    v.ViewTime = t; src.UpdatePipeline(t)
    if i == 0:
        import os as _o
        if _o.environ.get("VIS_VMAX"):
            lut.RescaleTransferFunction(0.0, float(_o.environ["VIS_VMAX"]))
        else:
            d.RescaleTransferFunctionToDataRange(False, True)
        print("colour range", lut.RGBPoints[0], lut.RGBPoints[-4])
    SaveScreenshot(f"{OUT}/f{i:04d}.png", v, ImageResolution=v.ViewSize)
    if i % 50 == 0: print("frame", i, flush=True)
print("done", len(sel), "->", OUT)
