"""Render a walled case as a section. pvbatch render_duct.py <caseDir> <outDir> [maxFrames]

The generic render_case.py draws the cut plane and nothing else. On injectorPipe that produced bands of
colour on black -- no wall anywhere in the frame, so nothing read as geometry. This renderer slices the
WALL PATCHES on the same plane as the field, which turns them into the section profile, and slices a
second set (VIS_HILITE) in a contrasting colour so a named region -- angledDuct's porous block -- is
visible rather than implied.

Env:
  VIS_FIELD   cell field to colour by            (default U; a vector is shown as magnitude)
  VIS_PRESET  colour preset                      (default "Viridis (matplotlib)")
  VIS_VMIN    colour range floor                 (default 0)
  VIS_VMAX    colour range ceiling               (default: the LAST step's range, not the first --
                                                  frame 0 is a start-up transient and auto-ranging on
                                                  it crushed every later injectorPipe frame)
  VIS_WALLS   patches for the section profile    (default: every patch whose name contains "wall")
  VIS_HILITE  patches to draw in the accent      (default none)
  VIS_SIZE    WxH                                (default 1920x1080)
  VIS_SCALE   camera half-height in model units  (default: an exact fit to the cut, plus 6%)
  VIS_ROLL    degrees to roll the view             (default 0; angledDuct's 135-degree dogleg leaves
                                                  half the frame empty axis-aligned, and -22.5 sets it
                                                  symmetric about the horizontal)
  VIS_LW      profile line width                 (default 4)
"""
import sys, os
from paraview.simple import *

CASE, OUT = sys.argv[1], sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(OUT, exist_ok=True)
foam = os.path.join(CASE, "case.foam")
open(foam, "a").close()

FIELD = os.environ.get("VIS_FIELD", "U")

probe = OpenFOAMReader(registrationName="probe", FileName=foam)
probe.UpdatePipeline()
available = list(probe.MeshRegions.Available)

def pick(var, default):
    raw = os.environ.get(var)
    if raw:
        want = [w.strip() for w in raw.split(",") if w.strip()]
        return [p for p in available if p in want or p.split("/")[-1] in want]
    return default

walls = pick("VIS_WALLS", [p for p in available if p.startswith("patch/") and "wall" in p.lower()])
hilite = pick("VIS_HILITE", [])
walls = [p for p in walls if p not in hilite]

fluid = OpenFOAMReader(registrationName="fluid", FileName=foam)
fluid.MeshRegions = ["internalMesh"]
fluid.CellArrays = [FIELD]
fluid.UpdatePipeline()
times = list(fluid.TimestepValues or [])
b = fluid.GetDataInformation().GetBounds()
span = [b[1]-b[0], b[3]-b[2], b[5]-b[4]]
thin = span.index(min(span))
ctr = [(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2]
print("timesteps", len(times), "bounds", b, "cut normal axis", thin, flush=True)
print("walls", walls, "hilite", hilite, flush=True)

def cut(inp, name):
    s = Slice(registrationName=name, Input=inp)
    s.SliceType.Origin = list(ctr)
    n = [0, 0, 0]; n[thin] = 1
    s.SliceType.Normal = n
    s.UpdatePipeline()
    return s

def patches(regions, name):
    if not regions:
        return None
    r = OpenFOAMReader(registrationName=name, FileName=foam)
    r.MeshRegions = regions
    r.CellArrays = []
    r.UpdatePipeline()
    return r

v = GetActiveViewOrCreate("RenderView")
_sz = os.environ.get("VIS_SIZE", "1920x1080").split("x")
v.ViewSize = [int(_sz[0]), int(_sz[1])]
v.OrientationAxesVisibility = 0
v.UseColorPaletteForBackground = 0
v.Background = [0.043, 0.055, 0.094]

flow = cut(fluid, "flow")
df = Show(flow, v)
df.Representation = "Surface"
# A scalar field has no "Magnitude" component, and asking for one silently leaves the mark unpainted.
_ai = flow.CellData.GetArray(FIELD)
_nc = _ai.GetNumberOfComponents() if _ai else 1
ColorBy(df, ("CELLS", FIELD, "Magnitude") if _nc > 1 else ("CELLS", FIELD))
lut = GetColorTransferFunction(FIELD)
lut.ApplyPreset(os.environ.get("VIS_PRESET", "Viridis (matplotlib)"), True)
df.SetScalarBarVisibility(v, False)

LW = float(os.environ.get("VIS_LW", "4"))
for regions, colour, width, name in ((walls,  [0.70, 0.76, 0.88], LW,       "rim"),
                                     (hilite, [0.98, 0.72, 0.30], LW * 1.6, "hot")):
    src = patches(regions, name + "src")
    if src is None:
        continue
    d = Show(cut(src, name), v)
    d.Representation = "Wireframe"
    d.AmbientColor = list(colour)
    d.DiffuseColor = list(colour)
    d.LineWidth = width

import math
import numpy as np
from paraview import servermanager as _sm
from paraview.numpy_support import vtk_to_numpy as _v2n

nrm = np.zeros(3); nrm[thin] = 1.0
up0 = np.array([0.0, 0.0, 1.0]) if thin != 2 else np.array([0.0, 1.0, 0.0])
# Which side of the cut the camera sits on decides the handedness of the picture, and it is not the
# same side for every normal: viewing a z-normal cut from -z puts +x on the LEFT, so angledDuct came
# out mirrored -- inlet on the right, outlet on the left. Pick the side that sends the remaining axis
# to screen-right, whichever axis was cut.
axr = np.array([1.0, 0.0, 0.0]) if thin != 0 else np.array([0.0, 1.0, 0.0])
side = 1.0
view = -side*nrm
if float(np.dot(np.cross(view, up0), axr)) < 0:
    side = -1.0
    view = -side*nrm
# Roll about the VIEW direction, so a given VIS_ROLL turns the picture the same way on either side.
roll = math.radians(float(os.environ.get("VIS_ROLL", "0")))
up = up0
if roll:
    up = up0*math.cos(roll) + np.cross(view, up0)*math.sin(roll)
    up /= np.linalg.norm(up)
right = np.cross(view, up)

# Fit the CUT, not the bounding box: an L-shaped duct fills its box only at the two legs, and once the
# view is rolled an axis-aligned box over-estimates the extent in both directions.
_mb = MergeBlocks(Input=flow)
_mb.UpdatePipeline(times[-1] if times else 0.0)
pts = _v2n(_sm.Fetch(_mb).GetPoints().GetData())
pu, pr = pts @ up, pts @ right
aspect = v.ViewSize[0] / float(v.ViewSize[1])
half = max((pu.max()-pu.min())*0.5, (pr.max()-pr.min())*0.5/aspect) * 1.06
mid = up*(pu.max()+pu.min())*0.5 + right*(pr.max()+pr.min())*0.5 + nrm*float(np.dot(ctr, nrm))

v.CameraFocalPoint = [float(x) for x in mid]
v.CameraPosition = [float(x) for x in (mid + side*nrm*max(span)*3)]
v.CameraViewUp = [float(x) for x in up]
v.CameraParallelProjection = 1
v.CameraParallelScale = float(os.environ.get("VIS_SCALE", half))

sel = times if not LIMIT else times[::max(1, len(times)//LIMIT)][:LIMIT]
# Range from the LAST step, never the first: frame 0 is a start-up transient whose peak is far above
# the converged field (injectorPipe: 33.7 m/s against 17.3), and ranging on it blacks out the clip.
flow.UpdatePipeline(sel[-1])
df.RescaleTransferFunctionToDataRange(False, True)
lo = float(os.environ.get("VIS_VMIN", lut.RGBPoints[0]))
hi = float(os.environ.get("VIS_VMAX", lut.RGBPoints[-4]))
lut.RescaleTransferFunction(lo, hi)
print("colour range", lo, hi, flush=True)

for i, t in enumerate(sel):
    v.ViewTime = t
    flow.UpdatePipeline(t)
    SaveScreenshot(f"{OUT}/f{i:04d}.png", v, ImageResolution=v.ViewSize)
    if i % 50 == 0:
        print("frame", i, flush=True)
print("done", len(sel), "->", OUT)
