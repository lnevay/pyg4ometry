import numpy as np
import pytest

from pyg4ometry import geant4


@pytest.mark.parametrize("scale", [None, [-1, 1, 1]])
def test_pv_extent_matches_placed_mesh(scale):
    # For a box the extent of the placed solid is exactly the bounds of its placed mesh, so
    # the two must agree for an arbitrary rotation and a reflection.
    reg = geant4.Registry()
    world = geant4.solid.Box("ws", 1000, 1000, 1000, reg)
    worldLV = geant4.LogicalVolume(world, "G4_Galactic", "wl", reg)
    box = geant4.solid.Box("bs", 100, 20, 10, reg)
    boxLV = geant4.LogicalVolume(box, "G4_Cu", "bl", reg)
    pv = geant4.PhysicalVolume([0.3, 0.5, 0.7], [10, 20, 30], boxLV, "bp", worldLV, reg, scale=scale)

    verts = np.array(worldLV._getPhysicalDaughterMesh(pv).toVerticesAndPolygons()[0])
    lo, hi = pv.extent()
    assert lo == pytest.approx(verts.min(axis=0))
    assert hi == pytest.approx(verts.max(axis=0))
