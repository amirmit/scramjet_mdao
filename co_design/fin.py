from utilities import Vector3, cross_product
from hypervehicle.geom.path import Line, Arc, Polyline
from hypervehicle.geom.compsurface.swept import SweptComponent
from hypervehicle import Component, logger
import numpy as np


class Fin(Component):
    r"""
    A really basic fin class.
    Has four sensitivity parameters for the thickness at the root and tip for
    the leading and trailing edges.

    The coordinates of the fin are specified by the reference parameters
    for the points at the base and wing tip of the leading and trailing edges.
    As we cant treat the Vector3 points as FloatWithSens, they are ref_params.

    a = 2c
    b = c

    Top view
        pLeadingEdgeBase (p0)
        o__
        |   --- ___
        |           --- _o pLeadingEdgeTip (p3)
        |                |
      a |                |
        |                | c
        |                |
        o________________o pTrailingEdgeTip (p2)
                 b     
        pTrailingEdgeBase
        (p1)


    Side view (It's two circles connected by tangent lines. That's tough to draw)

           ------- _______
         /                  \
        o                    o pTrailingEdgeBase
         \         ________ /
           -------
        pLeadingEdgeBase

    """

    def set_defaults(self) -> None:
        self.params = {
            "fin_area": 0.1,
            "L_rocket": 8,
            "length": 4,
            "D_rocket": 1.7,
        }
        self.ref_params = {
            "leading_thickness_base": 0.10,
            "trailing_thickness_base": 0.10,
            "leading_thickness_tip": 0.05,
            "trailing_thickness_tip": 0.05,
            "close": True,
            "orientation": 0,
            "fairing": 0,
        }
        self.mesh_params = {
            "mesh_resolution": 2,
            "n_span": 2,
        }
        return

    def geometry(self):
        def patch_generator(
            fin_area: float,
            L_rocket: float,
            D_rocket: float,
            length: float,
            fairing: float,
            orientation: float,
            leading_thickness_base: float,
            leading_thickness_tip: float,
            trailing_thickness_base: float,
            trailing_thickness_tip: float,
            mesh_resolution: float,
            n_span,
            close,
        ):
            # This enables everything except for parameters to be pre-defined
            # A patch generator should take in variables and output a list of patches in the patch_dict format
            c = (2 / 3 * fin_area) ** 0.5

            if fairing == 0:
                L_fairing = length + 1
            else:
                L_fairing = 1.5 * length

            p0 = Vector3(
                x=-(L_rocket + L_fairing) + 2 * c + trailing_thickness_base,
                y=(D_rocket / 2 - 0.05) * np.cos(orientation),
                z=(D_rocket / 2 - 0.05) * np.sin(orientation),
            )
            p1 = Vector3(
                x=-(L_rocket + L_fairing) + trailing_thickness_base,
                y=(D_rocket / 2 - 0.05) * np.cos(orientation),
                z=(D_rocket / 2 - 0.05) * np.sin(orientation),
            )
            p2 = Vector3(
                x=-(L_rocket + L_fairing) + trailing_thickness_base,
                y=(D_rocket / 2 + c) * np.cos(orientation),
                z=(D_rocket / 2 + c) * np.sin(orientation),
            )
            p3 = Vector3(
                x=-(L_rocket + L_fairing) + c + trailing_thickness_base,
                y=(D_rocket / 2 + c) * np.cos(orientation),
                z=(D_rocket / 2 + c) * np.sin(orientation),
            )

            base_paths = generate_paths(
                pa=p0,
                pb=p1,
                leading_thickness=leading_thickness_base,
                trailing_thickness=trailing_thickness_base,
                orientation=orientation,
            )
            tip_paths = generate_paths(
                pa=p3,
                pb=p2,
                leading_thickness=leading_thickness_tip,
                trailing_thickness=trailing_thickness_tip,
                orientation=orientation,
            )

            patches = SweptComponent(
                cross_sections=[base_paths, tip_paths],
                mesh_resolution={
                    "e0": mesh_resolution * 4,
                    "e1": mesh_resolution * 4,
                    "e2": mesh_resolution * 4,
                    "e3": mesh_resolution * 4,
                    "sweep": mesh_resolution * 4 * n_span,
                },
                close_ends=close,
            )

            patches.generate_patches()

            patch_list = []
            for i, key in enumerate(patches.patches.keys()):
                logger.debug(f" Adding Patch to patch_list[{i}]: label={key}; key={i}")
                patch_list += [
                    {
                        "patch_func": patches.patches[key],
                        "n_r": patches.patch_res_r[key],
                        "n_s": patches.patch_res_s[key],
                        "cf_list": None,
                        "keys": i,
                    }
                ]

            return patch_list

        self.make_surfaces_from_generator(
            "fin_patches",
            patch_generator,
            [
                "fin_area",
                "L_rocket",
                "D_rocket",
                "length",
                "fairing",
                "orientation",
                "leading_thickness_base",
                "leading_thickness_tip",
                "trailing_thickness_base",
                "trailing_thickness_tip",
                "mesh_resolution",
                "n_span",
                "close",
            ],
        )


def generate_paths(pa, pb, leading_thickness, trailing_thickness, orientation):
    lead_offset = Vector3(
        0,
        leading_thickness / 2 * np.sin(orientation),
        leading_thickness / 2 * np.cos(orientation),
    )
    trail_offset = Vector3(
        0,
        trailing_thickness / 2 * np.sin(orientation),
        trailing_thickness / 2 * np.cos(orientation),
    )
    e0 = Line(pa + lead_offset, pb + trail_offset)
    e2 = Line(pb - trail_offset, pa - lead_offset)

    e1 = Polyline(
        [
            Arc(pb + trail_offset, pb + Vector3(-trailing_thickness / 2, 0, 0), pb),
            Arc(pb + Vector3(-trailing_thickness / 2, 0, 0), pb - trail_offset, pb),
        ]
    )
    e3 = Polyline(
        [
            Arc(pa - lead_offset, pa + Vector3(leading_thickness / 2, 0, 0), pa),
            Arc(pa + Vector3(leading_thickness / 2, 0, 0), pa + lead_offset, pa),
        ]
    )

    return e0, e1, e2, e3
