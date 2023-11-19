import os
import pyg4ometry
import pyg4ometry.convert as convert
import pyg4ometry.visualisation as vi
from pyg4ometry.fluka import RPP, Region, Zone, FlukaRegistry, Material, Compound


def Test(vis=False, interactive=False):
    freg = FlukaRegistry()

    bkh = RPP("OUTER", -50, 50, -50, 50, -50, 50, flukaregistry=freg)
    box = RPP("RPP_BODY", -10, 10, -20, 20, -30, 30, flukaregistry=freg)


    bkhZone = Zone()
    bkhZone.addIntersection(bkh)
    bkhZone.addSubtraction(box)
    bkhRegion = Region("BLKBODY")
    bkhRegion.addZone(bkhZone)
    freg.addRegion(bkhRegion)
    
    zone = Zone()
    zone.addIntersection(box)
    boxRegion = Region("BOX")
    boxRegion.addZone(zone)
    freg.addRegion(boxRegion)

    freg.addMaterialAssignments("BLCKHOLE", bkhRegion)
    # here we don't define TUNGSTEN as it's a built in fluka material
    freg.addMaterialAssignments("TUNGSTEN", boxRegion)

    # test we have correctly identified it as builtin
    m = freg.materials[freg.assignmas[boxRegion.name]]
    print(m)
    print(type(m))
    assert(type(m) == pyg4ometry.fluka.material.BuiltIn)
    
    w = pyg4ometry.fluka.Writer()
    w.addDetector(freg)
    w.write("T807_material_builtin.inp")
    
    return {"testStatus": True, "logicalVolume": None, "vtkViewer": None}


if __name__ == "__main__":
    Test(True, True)
