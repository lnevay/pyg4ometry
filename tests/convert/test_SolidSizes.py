import numpy as np
import pytest

from pyg4ometry import convert, fluka, geant4
from pyg4ometry.geant4 import solid as s


def _meshExtent(solid):
    verts = np.array(solid.mesh().toVerticesAndPolygons()[0])
    return verts.min(axis=0), verts.max(axis=0)


def test_paraboloid_mesh_half_length():
    # As in Geant4 and GDML, pDz is a half-length, so the solid spans -pDz to +pDz with
    # radius pR1 at -pDz and pR2 at +pDz.
    solid = s.Paraboloid("p", 10, 5, 20, geant4.Registry())
    lo, hi = _meshExtent(solid)
    assert lo[2] == pytest.approx(-10)
    assert hi[2] == pytest.approx(10)
    assert hi[0] == pytest.approx(20, rel=1e-3)

    verts = np.array(solid.mesh().toVerticesAndPolygons()[0])
    bottom = verts[np.isclose(verts[:, 2], -10)]
    assert np.hypot(bottom[:, 0], bottom[:, 1]).max() == pytest.approx(5, rel=1e-3)


def test_paraboloid_to_fluka_half_length():
    # The end planes of the converted region must be at +-pDz, in cm.
    solid = s.Paraboloid("p", 100, 50, 200, geant4.Registry())
    freg = fluka.FlukaRegistry()
    convert.geant42Fluka.geant4Paraboloid2Fluka(0, solid, flukaRegistry=freg)
    planes = sorted(b.z for b in freg.bodyDict.values() if isinstance(b, fluka.XYP))
    assert planes == pytest.approx([-10, 10])


def test_generic_trap_mesh_reaches_top_face():
    # The top face of the mesh must be at the top vertices and not interpolated short of
    # them. Bottom is 10 x 10 mm, top is 20 x 20 mm, dz is a half-length of 15 mm.
    solid = s.GenericTrap(
        "g", -5, -5, -5, 5, 5, 5, 5, -5, -10, -10, -10, 10, 10, 10, 10, -10, 15, geant4.Registry()
    )
    verts = np.array(solid.mesh().toVerticesAndPolygons()[0])
    top = verts[np.isclose(verts[:, 2], 15)]
    bottom = verts[np.isclose(verts[:, 2], -15)]
    assert np.abs(top[:, :2]).max() == pytest.approx(10)
    assert np.abs(bottom[:, :2]).max() == pytest.approx(5)
    # Frustum of square cross-section, volume h / 3 * (A1 + A2 + sqrt(A1 A2)).
    assert solid.mesh().volume() == pytest.approx(30 / 3 * (100 + 400 + 200))


def test_generic_polyhedra_to_fluka_size():
    # Radii up to 80 mm and z up to 300 mm should give planes through points at up to
    # 8 cm in radius and 30 cm in z.
    solid = s.GenericPolyhedra("p", 0, 2 * np.pi, 6, [20, 80, 80, 20], [0, 0, 300, 300], geant4.Registry())
    freg = fluka.FlukaRegistry()
    convert.geant42Fluka.geant4Polyhedra2Fluka(0, solid, flukaRegistry=freg)
    points = np.array([list(b.point) for b in freg.bodyDict.values()])
    assert np.hypot(points[:, 0], points[:, 1]).max() == pytest.approx(8)
    assert np.abs(points[:, 2]).max() == pytest.approx(30)
