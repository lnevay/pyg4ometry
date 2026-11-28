import numpy as np
import pytest

import pyg4ometry.gdml as gd
import pyg4ometry.geant4 as g4
from pyg4ometry.geant4 import solid as s

pi = np.pi


def _tet(r):
    pts = [[0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 30]]
    return s.Tet("s", *[gd.Position(f"p{i}", *p, "mm", r) for i, p in enumerate(pts)], r)


def _tessellated(r):
    vertices = [[0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 30]]
    facets = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    return s.TessellatedSolid("s", [vertices, facets], r, s.TessellatedSolid.MeshType.Freecad)


def _boolean(cls):
    def make(r):
        b1 = s.Box("b1", 20, 20, 20, r)
        b2 = s.Box("b2", 10, 10, 10, r)
        return cls("s", b1, b2, [[0, 0, 0], [5, 5, 5]], r)

    return make


def _multiUnion(r):
    b1 = s.Box("b1", 20, 20, 20, r)
    b2 = s.Box("b2", 10, 10, 10, r)
    return s.MultiUnion("s", [b1, b2], [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [30, 0, 0]]], r)


# Each entry builds one solid in a fresh registry. Variants with phi segments, inner radii
# and cuts are included as these use separate meshing paths where normals can be flipped.
SOLIDS = {
    "Box": lambda r: s.Box("s", 10, 20, 30, r),
    "Tubs": lambda r: s.Tubs("s", 0, 10, 20, 0, 2 * pi, r),
    "Tubs_rmin_phi": lambda r: s.Tubs("s", 5, 10, 20, 0.2, 1.5 * pi, r),
    "CutTubs": lambda r: s.CutTubs("s", 0, 10, 20, 0, 2 * pi, [0, -0.7, -0.71], [0.7, 0, 0.71], r),
    "CutTubs_rmin_phi": lambda r: s.CutTubs("s", 5, 10, 20, 0.2, 1.5 * pi, [0, -0.7, -0.71], [0.7, 0, 0.71], r),
    "Cons": lambda r: s.Cons("s", 0, 10, 0, 5, 20, 0, 2 * pi, r),
    "Cons_rmin_phi": lambda r: s.Cons("s", 2, 10, 1, 5, 20, 0.2, 1.5 * pi, r),
    "Para": lambda r: s.Para("s", 10, 20, 30, 0.2, 0.3, 0.4, r),
    "Trd": lambda r: s.Trd("s", 10, 20, 30, 40, 50, r),
    "Trap": lambda r: s.Trap("s", 40, 0.1, 0.2, 10, 8, 12, 0.1, 20, 16, 24, 0.1, r),
    "Sphere": lambda r: s.Sphere("s", 0, 10, 0, 2 * pi, 0, pi, r),
    "Sphere_rmin_phi_theta": lambda r: s.Sphere("s", 5, 10, 0.2, 1.5 * pi, 0.3, 1.7, r),
    "Orb": lambda r: s.Orb("s", 10, r),
    "Torus": lambda r: s.Torus("s", 0, 5, 30, 0, 2 * pi, r),
    "Torus_rmin_phi": lambda r: s.Torus("s", 2, 5, 30, 0.2, 1.5 * pi, r),
    "Polycone": lambda r: s.Polycone("s", 0, 2 * pi, [0, 10, 30], [0, 0, 0], [5, 8, 8], r),
    "Polycone_rmin_phi": lambda r: s.Polycone("s", 0.2, 1.5 * pi, [0, 10, 30], [2, 2, 3], [5, 8, 8], r),
    "GenericPolycone": lambda r: s.GenericPolycone("s", 0, 2 * pi, [2, 8, 8, 2], [0, 0, 30, 30], r),
    "GenericPolycone_phi": lambda r: s.GenericPolycone("s", 0.2, 1.5 * pi, [2, 8, 8, 2], [0, 0, 30, 30], r),
    "Polyhedra": lambda r: s.Polyhedra("s", 0, 2 * pi, 6, 2, [0, 20], [0, 0], [10, 10], r),
    "Polyhedra_rmin_phi": lambda r: s.Polyhedra("s", 0.2, 1.5 * pi, 6, 2, [0, 20], [2, 3], [10, 10], r),
    "GenericPolyhedra": lambda r: s.GenericPolyhedra("s", 0, 2 * pi, 6, [2, 8, 8, 2], [0, 0, 30, 30], r),
    "GenericPolyhedra_phi": lambda r: s.GenericPolyhedra("s", 0.2, 1.5 * pi, 6, [2, 8, 8, 2], [0, 0, 30, 30], r),
    "EllipticalTube": lambda r: s.EllipticalTube("s", 10, 5, 15, r),
    "Ellipsoid": lambda r: s.Ellipsoid("s", 10, 5, 15, -100, 100, r),
    "Ellipsoid_cut": lambda r: s.Ellipsoid("s", 10, 5, 15, -5, 10, r),
    "EllipticalCone": lambda r: s.EllipticalCone("s", 0.5, 0.25, 20, 10, r),
    "Paraboloid": lambda r: s.Paraboloid("s", 10, 5, 10, r),
    "Hype": lambda r: s.Hype("s", 0, 10, 0, 0.5, 20, r),
    "Hype_inner": lambda r: s.Hype("s", 2, 10, 0.3, 0.5, 20, r),
    "Tet": _tet,
    "ExtrudedSolid": lambda r: s.ExtrudedSolid(
        "s", [[0, 0], [10, 0], [10, 20], [5, 10], [0, 20]], [[-15, [0, 0], 1], [15, [2, 1], 0.5]], r
    ),
    "TwistedBox": lambda r: s.TwistedBox("s", 0.5, 10, 20, 30, r),
    "TwistedTrd": lambda r: s.TwistedTrd("s", 0.5, 10, 20, 30, 40, 50, r),
    "TwistedTrap": lambda r: s.TwistedTrap("s", 0.5, 40, 0.1, 0.2, 10, 8, 12, 20, 16, 24, 0.1, r),
    "TwistedTubs": lambda r: s.TwistedTubs("s", 5, 10, 20, 1.5 * pi, 0.5, r),
    "GenericTrap": lambda r: s.GenericTrap("s", -5, -5, -5, 5, 5, 5, 5, -5, -10, -10, -10, 10, 10, 10, 10, -10, 15, r),
    "GenericTrap_twisted": lambda r: s.GenericTrap("s", -5, -5, -5, 5, 5, 5, 5, -5, -10, -8, -8, 10, 10, 8, 8, -10, 15, r),
    "TessellatedSolid": _tessellated,
    "Scaled": lambda r: s.Scaled("s", s.Orb("o", 10, r), 1, 2, 0.5, r),
    "Union": _boolean(s.Union),
    "Subtraction": _boolean(s.Subtraction),
    "Intersection": _boolean(s.Intersection),
    "MultiUnion": _multiUnion,
}


@pytest.mark.parametrize("name", SOLIDS)
def test_solid_mesh_volume_positive(name):
    # A negative volume means the mesh normals point inwards, which breaks booleans.
    solid = SOLIDS[name](g4.Registry())
    assert solid.mesh().volume() > 0
