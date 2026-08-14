#!/usr/bin/env python3
"""Create the lightweight rigid medicine-bottle USD used by the final demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def material(stage: Usd.Stage, path: str, color: tuple[float, float, float], roughness: float):
    output = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    output.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return output


def bind(prim: Usd.Prim, output: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(output)


def cylinder(
    stage: Usd.Stage,
    path: str,
    *,
    radius: float,
    height: float,
    z: float,
    output: UsdShade.Material,
    collision: bool,
) -> None:
    geom = UsdGeom.Cylinder.Define(stage, path)
    geom.CreateAxisAttr(UsdGeom.Tokens.z)
    geom.CreateRadiusAttr(radius)
    geom.CreateHeightAttr(height)
    geom.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
    bind(geom.GetPrim(), output)
    if collision:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())


def cube(
    stage: Usd.Stage,
    path: str,
    *,
    scale: tuple[float, float, float],
    position: tuple[float, float, float],
    output: UsdShade.Material,
) -> None:
    geom = UsdGeom.Cube.Define(stage, path)
    geom.CreateSizeAttr(1.0)
    geom.AddScaleOp().Set(Gf.Vec3f(*scale))
    geom.AddTranslateOp().Set(Gf.Vec3d(*position))
    bind(geom.GetPrim(), output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/medicine_bottle.usda"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(args.output.resolve()))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/MedicineBottle")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(0.050)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -0.004))

    amber = material(stage, "/Looks/AmberPlastic", (0.55, 0.16, 0.035), 0.32)
    white = material(stage, "/Looks/WhitePlastic", (0.92, 0.94, 0.96), 0.50)
    red = material(stage, "/Looks/MedicalRed", (0.78, 0.025, 0.035), 0.42)

    cylinder(
        stage,
        "/MedicineBottle/Body",
        radius=0.0235,
        height=0.088,
        z=0.0,
        output=amber,
        collision=True,
    )
    cylinder(
        stage,
        "/MedicineBottle/Cap",
        radius=0.025,
        height=0.020,
        z=0.054,
        output=white,
        collision=True,
    )
    # Slightly larger visual shell forms a clean pharmacy label without
    # changing the collision hull.
    cylinder(
        stage,
        "/MedicineBottle/Label",
        radius=0.0239,
        height=0.046,
        z=-0.002,
        output=white,
        collision=False,
    )
    # Red cross faces the hero/table cameras from the negative-Y side.
    cube(
        stage,
        "/MedicineBottle/MedicalCrossVertical",
        scale=(0.010, 0.0015, 0.027),
        position=(0.0, -0.0242, -0.002),
        output=red,
    )
    cube(
        stage,
        "/MedicineBottle/MedicalCrossHorizontal",
        scale=(0.024, 0.0015, 0.011),
        position=(0.0, -0.0243, -0.002),
        output=red,
    )

    stage.GetRootLayer().Save()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
