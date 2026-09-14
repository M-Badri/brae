"""Render gasMixing/injectorPipe as a cutaway. pvbatch render_injector.py <caseDir> <outDir> [maxFrames]

The generic renderer draws the cut plane and nothing else, which for this case produced bands of colour
on black: the first render was read as "a small square, but not the pipe itself". A pipe needs its wall
drawn, so this one slices the wall patches on the same plane to get a crisp profile line -- the narrow
Dia 0.10 inlet pipe, the fuel stub entering from +z at x 0.05-0.15, the step at x 0.25, the Dia 0.20 mixing
pipe -- and puts the far half-shell behind it for depth. The domain is 1.3 x 0.2 (6.5:1), so the frame is
cut to that aspect instead of leaving the flow as a ribbon inside 16:9.
"""
import sys, os
from paraview.simple import *

CASE, OUT = sys.argv[1], sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(OUT, exist_ok=True)
foam = os.path.join(CASE, "case.foam")
open(foam, "a").close()

WALLS = ["patch/walls_pipe_air", "patch/walls_pipe_fuel", "patch/walls_pipe_main"]

fluid = OpenFOAMReader(registrationName="fluid", FileName=foam)
fluid.MeshRegions = ["internalMesh"]
fluid.CellArrays = ["U", "p"]
fluid.UpdatePipeline()
times = list(fluid.TimestepValues or [])
b = fluid.GetDataInformation().GetBounds()
cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
print("timesteps", len(times), "bounds", b, flush=True)

# The mesh is thinnest in y, so y = 0 is the plane that shows the pipe along its axis and cuts the
# fuel stub down its middle.
def cut(inp, name):
    s = Slice(registrationName=name, Input=inp)
    s.SliceType.Origin = [cx, 0.0, cz]
    s.SliceType.Normal = [0, 1, 0]
    s.UpdatePipeline()
    return s

wall = OpenFOAMReader(registrationName="wall", FileName=foam)
wall.MeshRegions = WALLS
wall.CellArrays = []
wall.UpdatePipeline()

# Keep the half of the shell on the far side of the cut so the camera looks into an open pipe rather
# than at its outside.
shell = Clip(registrationName="shell", Input=wall)
shell.ClipType.Origin = [cx, 0.0, cz]
shell.ClipType.Normal = [0, 1, 0]
shell.Invert = 0
shell.UpdatePipeline()

flow = cut(fluid, "flow")
rim = cut(wall, "rim")

v = GetActiveViewOrCreate("RenderView")
_sz = os.environ.get("VIS_SIZE", "1920x360").split("x")
v.ViewSize = [int(_sz[0]), int(_sz[1])]
v.OrientationAxesVisibility = 0
v.UseColorPaletteForBackground = 0
v.Background = [0.043, 0.055, 0.094]

ds = Show(shell, v)
ds.Representation = "Surface"
ds.DiffuseColor = [0.13, 0.16, 0.24]
ds.Ambient = 0.55
ds.Diffuse = 0.45
ds.Specular = 0.0

df = Show(flow, v)
df.Representation = "Surface"
ColorBy(df, ("CELLS", "U", "Magnitude"))
lut = GetColorTransferFunction("U")
lut.ApplyPreset("Viridis (matplotlib)", True)
df.SetScalarBarVisibility(v, False)

# The profile line is what makes it read as a pipe, so it is drawn last, thick enough to read as the
# sectioned wall rather than as a hairline: 6 px at 1920x360 is about the 0.008 wall the drawing implies.
dr = Show(rim, v)
dr.Representation = "Wireframe"
dr.AmbientColor = [0.70, 0.76, 0.88]
dr.DiffuseColor = [0.70, 0.76, 0.88]
dr.LineWidth = float(os.environ.get("VIS_LW", "6"))

v.CameraPosition = [cx, cy - max(b[1]-b[0], 1.0)*3, cz]
v.CameraFocalPoint = [cx, cy, cz]
v.CameraViewUp = [0, 0, 1]
v.CameraParallelProjection = 1
# Half-height in model units. 0.125 leaves the Dia 0.20 pipe at 80% of a 1920x360 frame -- margin for the
# sectioned wall band -- while 0.125 * 1920/360 = 1.33 still spans the full 1.3 length.
v.CameraParallelScale = float(os.environ.get("VIS_SCALE", "0.125"))
for k, i in (("VIS_CX", 0), ("VIS_CY", 1), ("VIS_CZ", 2)):
    if os.environ.get(k):
        v.CameraFocalPoint[i] = float(os.environ[k])
        v.CameraPosition[i] = float(os.environ[k]) + (v.CameraPosition[i] - v.CameraFocalPoint[i])

sel = times if not LIMIT else times[::max(1, len(times)//LIMIT)][:LIMIT]
for i, t in enumerate(sel):
    v.ViewTime = t
    flow.UpdatePipeline(t)
    if i == 0:
        # Fixed range, never frame-0 auto-range: the first step is a start-up transient peaking at
        # 33.7 m/s while the converged field reaches 17.3, so auto-ranging crushed every later frame
        # into the dark end of the ramp. 16 is just above the p99.5 of the converged slice.
        lut.RescaleTransferFunction(0.0, float(os.environ.get("VIS_VMAX", "16")))
        print("colour range", lut.RGBPoints[0], lut.RGBPoints[-4], flush=True)
    SaveScreenshot(f"{OUT}/f{i:04d}.png", v, ImageResolution=v.ViewSize)
    if i % 50 == 0:
        print("frame", i, flush=True)
print("done", len(sel), "->", OUT)
