import numpy as _np
import pyg4ometry as _pyg4

def beamPipe(stlFileName, feature1 = -1, feature2 = -1, planeAngles = [[0,0,0]], vis=True, interactive=True) :

    datFileName = stlFileName.replace("stl", "dat")

    if feature1 == -1 :
        vis = True
        interactive = True

    v = _pyg4.features.extract(stlFileName,
                               angle = 46,
                               circumference=2*_np.pi*8,
                               planes=[],
                               outputFileName=datFileName,
                               bViewer=vis,
                               bViewerInteractive=interactive)

    fd = _pyg4.features.algos.FeatureData()
    fd.readFile(datFileName)

    if feature1 == -1 :
        return

    p1 = fd.features[feature1]["plane"]
    p2 = fd.features[feature2]["plane"]

    pp1 = _pyg4.features.Plane(p1[0:3], p1[3:])
    pp2 = _pyg4.features.Plane(p2[0:3], p2[3:])
    pp3 = _pyg4.features.Plane([0, 0, 0], [0, 0, 1])

    cs = _pyg4.features.CoordinateSystem()
    cs.makeFromPlanes(pp1, pp2, pp3)

    csa  = [cs.coordinateSystem(ang[0],ang[1],ang[2]) for ang in planeAngles]

    v.addPlane(cs.origin, cs.e1, cs.e2, cs.dist)
    v.addAxis(cs.origin,[cs.dist,cs.dist,cs.dist],cs.rot,label=True,disableCone=True)
    v.view(interactive=True)

    v = _pyg4.features.extract(stlFileName,
                               angle = 46,
                               circumference=2*_np.pi*8,
                               planes=csa,
                               outputFileName=datFileName,
                               bViewer=True,
                               bViewerInteractive=True)