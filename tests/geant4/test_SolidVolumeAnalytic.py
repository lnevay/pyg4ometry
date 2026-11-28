import numpy as np
import pytest

import pyg4ometry.gdml as gd
import pyg4ometry.geant4 as g4
from pyg4ometry.geant4 import solid as s

pi = np.pi

# Relative tolerances. Solids with only flat faces should be exact. Curved solids are faceted with
# a modest nslice / nstack, so their meshes are inscribed and a few percent smaller.
exact = 1e-6
curved = 0.04


def _shoelace(pts):
    x, y = np.array(pts).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _simpson(length, a1, am, a2):
    # Exact when the cross-sectional area is quadratic in z, as for all linearly interpolated solids.
    return length / 6 * (a1 + 4 * am + a2)


def _frustum(h, r1, r2):
    # Volume per radian of a solid of revolution with radius varying linearly from r1 to r2.
    return h / 6 * (r1**2 + r1 * r2 + r2**2)


def _tet(r):
    pts = [[0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 30]]
    return s.Tet("s", *[gd.Position(f"p{i}", *p, "mm", r) for i, p in enumerate(pts)], r)


def _tessellated(r):
    vertices = [[0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 30]]
    facets = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    return s.TessellatedSolid("s", [vertices, facets], r, s.TessellatedSolid.MeshType.Freecad)


def _boolean(cls):
    # Box 20 mm cube with a 10 mm cube offset by 10 mm in x, so they overlap by 5 x 10 x 10 mm.
    def make(r):
        b1 = s.Box("b1", 20, 20, 20, r)
        b2 = s.Box("b2", 10, 10, 10, r)
        return cls("s", b1, b2, [[0, 0, 0], [10, 0, 0]], r)

    return make


def _multiUnion(r):
    b1 = s.Box("b1", 20, 20, 20, r)
    b2 = s.Box("b2", 10, 10, 10, r)
    return s.MultiUnion("s", [b1, b2], [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [30, 0, 0]]], r)


def _genericTrapVolume(bot, top, length):
    mid = [(np.array(b) + np.array(t)) / 2 for b, t in zip(bot, top)]
    return _simpson(length, _shoelace(bot), _shoelace(mid), _shoelace(top))


def _hypeVolume(rIn, rOut, stIn, stOut, length):
    # Integral of pi (r_out(z)^2 - r_in(z)^2) over z, with r(z)^2 = r^2 + (z tan(stereo))^2.
    h = length / 2
    return 2 * pi * ((rOut**2 - rIn**2) * h + (np.tan(stOut) ** 2 - np.tan(stIn) ** 2) * h**3 / 3)


def _ellipsoidVolume(a, b, c, zBottom, zTop):
    return pi * a * b * ((zTop - zBottom) - (zTop**3 - zBottom**3) / (3 * c**2))


def _polyhedraVolume(numSide, dPhi, rzArea, tangent):
    # A regular polygon sector of corner radius R has area 0.5 R^2 sin(alpha), alpha = dPhi / numSide.
    # rzArea is the integral of r dr dz over the r-z cross-section. Tangent distances are converted
    # to corner radii as in G4Polyhedra.
    alpha = dPhi / numSide
    factor = 1 / np.cos(alpha / 2) ** 2 if tangent else 1
    return numSide * np.sin(alpha) * factor * rzArea


# Each entry is (builder, analytic volume in mm^3, relative tolerance). Lengths follow the GDML
# conventions documented in each solid class.
cases = {
    "Box": (lambda r: s.Box("s", 10, 20, 30, r), 10 * 20 * 30, exact),
    
    "Tubs": (lambda r: s.Tubs("s", 0, 10, 20, 0, 2 * pi, r), pi * 10**2 * 20, curved),
    
    "Tubs_rmin_phi": (lambda r: s.Tubs("s", 5, 10, 20, 0.2, 1.5 * pi, r), 1.5 * pi / 2 * (10**2 - 5**2) * 20, curved),
    
    # Cut planes pass through the centres of the end faces, so the volume equals the uncut tube.
    "CutTubs": (
        lambda r: s.CutTubs("s", 0, 10, 20, 0, 2 * pi, [0, -0.7, -0.71], [0.7, 0, 0.71], r),
        pi * 10**2 * 20,
        curved,
    ),
    "CutTubs_rmin": (
        lambda r: s.CutTubs("s", 5, 10, 20, 0, 2 * pi, [0, -0.7, -0.71], [0.7, 0, 0.71], r),
        pi * (10**2 - 5**2) * 20,
        curved,
    ),
    
    "Cons": (lambda r: s.Cons("s", 0, 10, 0, 5, 20, 0, 2 * pi, r), 2 * pi * _frustum(20, 10, 5), curved),
    "Cons_rmin_phi": (
        lambda r: s.Cons("s", 2, 10, 1, 5, 20, 0.2, 1.5 * pi, r),
        1.5 * pi * (_frustum(20, 10, 5) - _frustum(20, 2, 1)),
        curved,
    ),
    
    # Shearing does not change the volume.
    "Para": (lambda r: s.Para("s", 10, 20, 30, 0.2, 0.3, 0.4, r), 10 * 20 * 30, exact),
    
    "Trd": (lambda r: s.Trd("s", 10, 20, 30, 40, 50, r), _simpson(50, 10 * 30, 15 * 35, 20 * 40), exact),
    
    # GDML full lengths: z = 40, y1 = 10, x1 = 8, x2 = 12, y2 = 20, x3 = 16, x4 = 24.
    "Trap": (
        lambda r: s.Trap("s", 40, 0.1, 0.2, 10, 8, 12, 0.1, 20, 16, 24, 0.1, r),
        _simpson(40, 10 * (8 + 12) / 2, 15 * (12 + 18) / 2, 20 * (16 + 24) / 2),
        exact,
    ),
    
    "Sphere": (lambda r: s.Sphere("s", 0, 10, 0, 2 * pi, 0, pi, r, nslice=20, nstack=20), 4 / 3 * pi * 10**3, curved),
    "Sphere_rmin_phi_theta": (
        lambda r: s.Sphere("s", 5, 10, 0.2, 1.5 * pi, 0.3, 1.4, r, nslice=20, nstack=20),
        1.5 * pi / 3 * (10**3 - 5**3) * (np.cos(0.3) - np.cos(1.7)),
        curved,
    ),
    
    "Orb": (lambda r: s.Orb("s", 10, r), 4 / 3 * pi * 10**3, curved)
    ,
    "Torus": (lambda r: s.Torus("s", 0, 5, 30, 0, 2 * pi, r, nstack=20), 2 * pi**2 * 30 * 5**2, curved),
    "Torus_rmin_phi": (
        lambda r: s.Torus("s", 2, 5, 30, 0.2, 1.5 * pi, r, nstack=20),
        1.5 * pi * 30 * pi * (5**2 - 2**2),
        curved,
    ),
    
    "Polycone": (
        lambda r: s.Polycone("s", 0, 2 * pi, [0, 10, 30], [0, 0, 0], [5, 8, 8], r),
        2 * pi * (_frustum(10, 5, 8) + _frustum(20, 8, 8)),
        curved,
    ),
    
    "Polycone_rmin_phi": (
        lambda r: s.Polycone("s", 0.2, 1.5 * pi, [0, 10, 30], [2, 2, 3], [5, 8, 8], r),
        1.5 * pi * (_frustum(10, 5, 8) + _frustum(20, 8, 8) - _frustum(10, 2, 2) - _frustum(20, 2, 3)),
        curved,
    ),
    
    "GenericPolycone": (
        lambda r: s.GenericPolycone("s", 0, 2 * pi, [2, 8, 8, 2], [0, 0, 30, 30], r),
        pi * (8**2 - 2**2) * 30,
        curved,
    ),
    
    "GenericPolycone_phi": (
        lambda r: s.GenericPolycone("s", 0.2, 1.5 * pi, [2, 8, 8, 2], [0, 0, 30, 30], r),
        1.5 * pi / 2 * (8**2 - 2**2) * 30,
        curved,
    ),
    
    # Polyhedra radii are tangent distances, GenericPolyhedra radii are corner radii.
    "Polyhedra": (
        lambda r: s.Polyhedra("s", 0, 2 * pi, 6, 2, [0, 20], [0, 0], [10, 10], r),
        _polyhedraVolume(6, 2 * pi, 20 * 10**2 / 2, True),
        exact,
    ),
    
    "Polyhedra_rmin_phi": (
        lambda r: s.Polyhedra("s", 0.2, 1.5 * pi, 6, 2, [0, 20], [2, 3], [10, 10], r),
        _polyhedraVolume(6, 1.5 * pi, 20 * 10**2 / 2 - _frustum(20, 2, 3), True),
        exact,
    ),
    
    "GenericPolyhedra": (
        lambda r: s.GenericPolyhedra("s", 0, 2 * pi, 6, [2, 8, 8, 2], [0, 0, 30, 30], r),
        _polyhedraVolume(6, 2 * pi, 30 * (8**2 - 2**2) / 2, False),
        exact,
    ),
    
    "GenericPolyhedra_phi": (
        lambda r: s.GenericPolyhedra("s", 0.2, 1.5 * pi, 6, [2, 8, 8, 2], [0, 0, 30, 30], r),
        _polyhedraVolume(6, 1.5 * pi, 30 * (8**2 - 2**2) / 2, False),
        exact,
    ),
    
    "EllipticalTube": (lambda r: s.EllipticalTube("s", 10, 5, 15, r), pi * 10 * 5 * 30, curved),
    
    "Ellipsoid": (lambda r: s.Ellipsoid("s", 10, 5, 15, -100, 100, r, nslice=20, nstack=20), 4 / 3 * pi * 10 * 5 * 15, curved),
    
    "Ellipsoid_cut": (lambda r: s.Ellipsoid("s", 10, 5, 15, -5, 10, r, nslice=20, nstack=20), _ellipsoidVolume(10, 5, 15, -5, 10), curved),

# Semi-axes scale with the distance from the apex at z = zMax.
    "EllipticalCone": (
        lambda r: s.EllipticalCone("s", 0.5, 0.25, 20, 10, r),
        pi * 0.5 * 0.25 * ((20 + 10) ** 3 - (20 - 10) ** 3) / 3,
        curved,
    ),
    
    "Paraboloid": (lambda r: s.Paraboloid("s", 10, 5, 10, r), pi * 10 * (5**2 + 10**2), curved),
    
    "Hype": (lambda r: s.Hype("s", 0, 10, 0, 0.5, 20, r), _hypeVolume(0, 10, 0, 0.5, 20), curved),
    
    "Hype_inner": (lambda r: s.Hype("s", 2, 10, 0.3, 0.5, 20, r), _hypeVolume(2, 10, 0.3, 0.5, 20), curved),
    
    "Tet": (_tet, 10 * 20 * 30 / 6, exact),
    
    "TessellatedSolid": (_tessellated, 10 * 20 * 30 / 6, exact),
    
    # Offsets do not change the volume, scale factors vary linearly from 1 to 0.5.
    "ExtrudedSolid": (
        lambda r: s.ExtrudedSolid(
            "s", [[0, 0], [10, 0], [10, 20], [5, 10], [0, 20]], [[-15, [0, 0], 1], [15, [2, 1], 0.5]], r
        ),
        _shoelace([[0, 0], [10, 0], [10, 20], [5, 10], [0, 20]]) * _simpson(30, 1, 0.75**2, 0.5**2),
        exact,
    ),
    
    # Twisting rotates each cross-section without changing its area. The side faces are curved.
    "TwistedBox": (lambda r: s.TwistedBox("s", 0.5, 10, 20, 30, r), 10 * 20 * 30, curved),
    
    "TwistedTrd": (
        lambda r: s.TwistedTrd("s", 0.5, 10, 20, 30, 40, 50, r),
        _simpson(50, 10 * 30, 15 * 35, 20 * 40),
        curved,
    ),
    
    # GDML full lengths: dz = 40, dy1 = 10, dx1 = 8, dx2 = 12, dy2 = 20, dx3 = 16, dx4 = 24.
    "TwistedTrap": (
        lambda r: s.TwistedTrap("s", 0.5, 40, 0.1, 0.2, 10, 8, 12, 20, 16, 24, 0.1, r),
        _simpson(40, 10 * (8 + 12) / 2, 15 * (12 + 18) / 2, 20 * (16 + 24) / 2),
        curved,
    ),
    
    # dz is the half length. Vertices 1-4 are at -dz and 5-8 at +dz.
    "GenericTrap": (
        lambda r: s.GenericTrap("s", -5, -5, -5, 5, 5, 5, 5, -5, -10, -10, -10, 10, 10, 10, 10, -10, 15, r),
        _genericTrapVolume([[-5, -5], [-5, 5], [5, 5], [5, -5]], [[-10, -10], [-10, 10], [10, 10], [10, -10]], 30),
        exact,
    ),
    
    "GenericTrap_twisted": (
        lambda r: s.GenericTrap("s", -5, -5, -5, 5, 5, 5, 5, -5, -10, -8, -8, 10, 10, 8, 8, -10, 15, r),
        _genericTrapVolume([[-5, -5], [-5, 5], [5, 5], [5, -5]], [[-10, -8], [-8, 10], [10, 8], [8, -10]], 30),
        curved,
    ),
    
    "Scaled": (lambda r: s.Scaled("s", s.Orb("o", 10, r), 1, 2, 0.5, r), 4 / 3 * pi * 10**3, curved),
    
    "Union": (_boolean(s.Union), 20**3 + 10**3 - 5 * 10 * 10, exact),
    
    "Subtraction": (_boolean(s.Subtraction), 20**3 - 5 * 10 * 10, exact),
    
    "Intersection": (_boolean(s.Intersection), 5 * 10 * 10, exact),
    
    "MultiUnion": (_multiUnion, 20**3 + 10**3, exact),
}


@pytest.mark.parametrize("name", cases)
def test_solid_mesh_volume_analytic(name):
    build, volume, rtol = cases[name]
    solid = build(g4.Registry())
    assert solid.mesh().volume() == pytest.approx(volume, rel=rtol)
