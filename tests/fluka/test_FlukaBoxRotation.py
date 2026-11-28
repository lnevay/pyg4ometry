import numpy as np
import pytest

from pyg4ometry import convert, fluka


def _convertBox(body, freg):
    zone = fluka.Zone()
    zone.addIntersection(body)
    region = fluka.Region("REG")
    region.addZone(zone)
    freg.addRegion(region)
    freg.assignma("COPPER", region)
    return convert.fluka2Geant4(freg).getWorldVolume()


@pytest.mark.parametrize("leftHanded", [False, True])
def test_box_rotated_edges_geant4_extent(leftHanded):
    # A 10 x 2 x 1 box turned 30 deg about z. The converted solid must cover the turned box
    # and not an axis-aligned one. 30 deg rather than 45 deg so that turning the wrong way
    # gives a different extent.
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    vertex = np.array([0, 0, 1] if leftHanded else [0, 0, 0])
    e1 = np.array([10 * c, 10 * s, 0])
    e2 = np.array([-2 * s, 2 * c, 0])
    e3 = np.array([0, 0, -1] if leftHanded else [0, 0, 1])
    freg = fluka.FlukaRegistry()
    body = fluka.BOX("B", vertex, e1, e2, e3, flukaregistry=freg)

    lo, hi = _convertBox(body, freg).extent()
    corners = np.array(
        [vertex + i * e1 + j * e2 + k * e3 for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    )
    # Tolerance allows for the length safety applied in the conversion.
    assert lo == pytest.approx(corners.min(axis=0), abs=1e-5)
    assert hi == pytest.approx(corners.max(axis=0), abs=1e-5)


def test_box_rotation_is_proper():
    # Left-handed edges must still give a rotation and not a reflection.
    freg = fluka.FlukaRegistry()
    body = fluka.BOX("B", [0, 0, 0], [0, 0, 10], [0, 10, 0], [10, 0, 0], flukaregistry=freg)
    rot = body.rotation()
    assert np.linalg.det(rot) == pytest.approx(1)
    assert rot @ [1, 0, 0] == pytest.approx([0, 0, 1])
    assert rot @ [0, 1, 0] == pytest.approx([0, 1, 0])
