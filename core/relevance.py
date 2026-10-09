"""
Context-dependent relevance of preset fields.

Many traits only matter in some contexts: rib depth means nothing for an Opuntia (cladodes), the corolla
tube nothing for a sunflower head, spine length nothing when a cactus has no central spines. Each rule
here says when a field APPLIES, as a Python expression over the other fields of the same section, plus a
short note for the UI and the documentation.

Used by: the Blender panels (inactive fields are greyed out, sections with nothing applicable are hidden),
the JSON Schema / field reference ("x-applies-when"), and a perturbation test that changes every
non-applicable field and checks that the generated geometry stays identical.

Sections: Tree -> architecture, roots, bark, leaf_morphology; Cactus / Rosette -> profile;
Flower -> flower, infl; Vine and Vegetable -> profile, leaf; Fruit -> fruit; Grass -> profile. Enum fields compare by value (e.g. habit == 'Cladode').
"""

from __future__ import annotations

from enum import Enum

_NOT_CLADODE = ("habit != 'Cladode'", "Not used by cladodes (Opuntia)")
_ARMS = ("habit != 'Cladode' and arm_count > 0", "Only when the stem has arms")
_NOT_HEAD = ("not capitulum", "Not used by a capitulum (Asteraceae head)")
_HEAD = ("capitulum", "Only for a capitulum (Asteraceae head)")
_WOODY_ROOTS = ("system not in ('Fibrous', 'Tuberous', 'Tuberous cluster')", "Only for woody root systems")
_FIBROUS_ROOTS = ("system in ('Fibrous', 'Tuberous', 'Tuberous cluster')",
                  "Only for fibrous roots (also around storage roots)")
_STORAGE_ROOTS = ("system in ('Tuberous', 'Tuberous cluster')", "Only for storage roots")
_NOT_CORNER = ("model != 'Corner'", "Not used by unbranched Corner-model plants (palms)")

RULES: dict = {
    "Tree": {
        "architecture": {
            **{f: _NOT_CORNER for f in (
                "max_order", "branch_angle_mean_deg", "branch_angle_std_deg", "twig_angle_mean_deg",
                "branch_frequency_per_meter", "internode_length_base_m", "internode_decay_per_order",
                "gravitropism", "phototropism", "plagiotropy", "phyllotaxis", "whorl_size",
                "crown_widest_position", "crown_fullness")},
            "divergence_angle_deg": ("model != 'Corner' and phyllotaxis == 'Spiral'",
                                     "Only for spiral phyllotaxis"),
        },
        "roots": {
            **{f: _WOODY_ROOTS for f in (
                "lateral_count", "spread_crown_ratio", "max_depth_m", "beta", "taproot_share", "zrt_dbh_ratio",
                "sinker_spacing_m", "branch_spacing_m", "surface_exposure", "plank", "buttress_height_dbh",
                "tortuosity", "knees", "deep_roots", "pneumatophores")},
            **{f: _FIBROUS_ROOTS for f in ("fibrous_count", "fibrous_radius_m", "fibrous_spread_m")},
            **{f: _STORAGE_ROOTS for f in ("tuber_length_m", "tuber_radius_m")},
            "tuber_count": ("system == 'Tuberous cluster'", "Only for a tuberous cluster"),
            "heart_roots": ("system == 'Heart' and taproot_share > 0", "Only for heart root systems"),
            "stilt_height_dbh": ("system == 'Stilt'", "Only for stilt (prop) roots"),
            "fine_orders": ("fine_roots > 0", "Only with fine roots"),
        },
        "bark": {
            "blockiness": ("pattern in ('Fissured', 'Plated', 'Annulated')", "Only for fissured, plated or annulated bark"),
            "plate_tilt": ("pattern in ('Fissured', 'Plated', 'Annulated')", "Only for fissured, plated or annulated bark"),
            "segments": ("pattern in ('Fissured', 'Plated', 'Fibrous')", "Only for fissured, plated or fibrous bark"),
        },
        "leaf_morphology": {
            **{f: ("lobe_type != 'None'", "Only for lobed leaves") for f in (
                "lobe_count", "lobe_depth", "lobe_angle_deg", "lobe_width", "lobe_roundness", "lobe_apex_angle_deg")},
            **{f: ("margin_type != 'Entire'", "Only for toothed margins") for f in (
                "teeth_count", "tooth_height_ratio", "tooth_skew")},
            **{f: ("compound_type != 'Simple'", "Only for compound leaves") for f in (
                "leaflet_count", "leaflet_size_gradient")},
            "leaflet_angle_deg": ("compound_type in ('Pinnate', 'Spray')", "Only for pinnate compound leaves"),
            "rachis_length_ratio": ("compound_type in ('Pinnate', 'Spray')", "Only for pinnate compound leaves"),
            "terminal_leaflet": ("compound_type == 'Pinnate'", "Only for pinnate compound leaves"),
        },
    },
    "Cactus": {
        "profile": {
            **{f: _NOT_CLADODE for f in (
                "height_m", "diameter_m", "base_taper", "apex_dome", "apex_roundness", "apex_depression",
                "arrangement", "rib_count", "rib_twist_deg_per_m", "tubercle_height", "arm_count", "offsets")},
            "rib_depth": ("habit != 'Cladode' and arrangement == 'Ribs'", "Only for ribbed stems"),
            "rib_sharpness": ("habit != 'Cladode' and arrangement == 'Ribs'", "Only for ribbed stems"),
            **{f: _ARMS for f in ("arm_height_min", "arm_height_max", "arm_radius_ratio", "arm_reach_m",
                                  "arm_length_ratio", "arm_lean_deg", "arm_branching", "crown_fill")},
            "offset_scale": ("habit != 'Cladode' and offsets > 0", "Only when there are basal offsets"),
            **{f: ("habit == 'Cladode'", "Only for cladodes (Opuntia)") for f in (
                "pad_length_cm", "pad_width_ratio", "pad_thickness_ratio", "pad_levels", "pad_branching")},
            "central_length_cm": ("central_spines > 0", "Only when there are central spines"),
            "central_hook": ("central_spines > 0", "Only when there are central spines"),
            "tuber_length_cm": ("root_system in ('Tuberous', 'Tuberous cluster')", "Only for storage roots"),
            "tuber_radius_ratio": ("root_system in ('Tuberous', 'Tuberous cluster')", "Only for storage roots"),
            **{f: ("browning_height_m > 0", "Only when there is epidermal browning") for f in (
                "equator_bias", "equator_azimuth_deg", "scaling_color", "barking_color")},
            "scar_color": ("scars > 0", "Only when there are scars"),
        },
    },
    "Rosette": {
        "profile": {
            "teeth_size_cm": ("teeth_count > 0", "Only when the leaves have marginal teeth"),
            "teeth_hook": ("teeth_count > 0", "Only when the leaves have marginal teeth"),
            "offset_scale": ("offsets > 0", "Only when there are offsets"),
            "stem_scars": ("stem_height_m > 0", "Only for a visible stem"),
            "scar_spacing_mm": ("stem_height_m > 0", "Only for a visible stem"),
            "dead_color": ("dead_leaves > 0", "Only when there are withered leaves"),
            "blush_color": ("blush_amount > 0", "Only when there is a blush"),
            "blush_tip": ("blush_amount > 0", "Only when there is a blush"),
        },
    },
    "Flower": {
        "flower": {
            **{f: _HEAD for f in ("disc_florets", "disc_radius_cm", "disc_dome", "ray_count", "ray_fill",
                                  "involucre_bracts", "disc_color", "disc_center_color")},
            **{f: _NOT_HEAD for f in (
                "arrangement", "merosity", "zygomorphy", "petals_visible", "tube_length_cm", "sepal_length_ratio",
                "hypanthium_cm", "stamen_count", "style_length_ratio", "receptacle_height_cm")},
            "petal_whorls": ("not capitulum and arrangement == 'Whorled'", "Only for whorled flowers"),
            "spiral_tepals": ("not capitulum and arrangement == 'Spiral'", "Only for spiral flowers"),
            "lip_bias": ("not capitulum and zygomorphy > 0", "Only for zygomorphic flowers"),
            **{f: ("not capitulum and tube_length_cm > 0", "Only for a corolla tube") for f in (
                "tube_radius_cm", "tube_flare", "limb_fusion")},
            "hypanthium_scales": ("not capitulum and hypanthium_cm > 0", "Only with a floral cup / pericarpel"),
            **{f: ("not capitulum and sepal_length_ratio > 0", "Only with visible sepals") for f in (
                "sepal_aspect", "sepal_opening_deg")},
            **{f: ("not capitulum and stamen_count > 0", "Only when there are stamens") for f in (
                "stamen_length_ratio", "stamen_spread_deg", "anther_size_mm", "staminal_column",
                "stamen_declination")},
            **{f: ("not capitulum and style_length_ratio > 0", "Only with a visible pistil") for f in (
                "carpels", "stigma_size_mm")},
            "fringe_count": ("fringe > 0", "Only for fringed petals"),
            "guide_contrast": ("guide_lines > 0", "Only with nectar guides"),
            "spot_color": ("spots > 0", "Only for spotted petals"),
            "outer_color": ("outer_tint > 0", "Only with an outer-tepal tint"),
        },
        "infl": {
            "flower_count": ("kind != 'Solitary'", "Not for a solitary flower"),
            "maturation": ("kind != 'Solitary'", "Not for a solitary flower"),
            "pedicel_angle_deg": ("kind != 'Solitary'", "Not for a solitary flower"),
            "rachis_cm": ("kind not in ('Solitary', 'Umbel')", "Not for solitary flowers or umbels"),
            "pedicel_cm": ("kind not in ('Solitary', 'Spike', 'Catkin')", "Not for solitary flowers, spikes or catkins"),
            "divergence_deg": ("kind not in ('Solitary', 'Umbel')", "Not for solitary flowers or umbels"),
            **{f: ("kind == 'Panicle'", "Only for panicles") for f in (
                "branches", "branch_start", "branch_length_ratio", "branch_angle_deg", "branch_umbels")},
        },
    },
}

_TENDRILS = ("tendril_mode != 'None'", "Only when the plant has tendrils")
RULES["Vine"] = {
    "profile": {
        **{f: ("mode == 'Twining'", "Only for twining stems") for f in (
            "chirality", "coil_radius_cm", "coil_pitch_cm")},
        "tip_hook": ("tip_length_cm > 0", "Only with a free shoot tip"),
        **{f: ("branch_probability > 0", "Only when nodes bear lateral shoots") for f in (
            "branch_length_cm", "branch_angle_deg", "branch_droop")},
        **{f: _TENDRILS for f in ("tendril_length_cm", "tendril_branches", "tendril_coil_mm",
                                  "tendril_radius_mm", "tendril_reach")},
        "tendril_coils": ("tendril_mode != 'None' and tendril_reach < 1", "Only for free (uncaught) tendrils"),
        "rootlet_length_cm": ("aerial_roots > 0", "Only with nodal / adventitious roots"),
        "stem_color_old": ("woodiness > 0", "Only for a lignified stem base"),
    },
    "leaf": dict(RULES["Tree"]["leaf_morphology"]),
}

_BUNCH = ("cluster_berries > 1", "Only for bunches (grapes, cherries)")
_CROWN = ("crown_lobes > 0", "Only with a persistent calyx")
RULES["Fruit"] = {
    "fruit": {
        **{f: _BUNCH for f in ("cluster_length_cm", "cluster_width_cm", "shoulders", "compactness", "pedicel_cm",
                               "peduncle_cm")},
        **{f: _CROWN for f in ("crown_length_cm", "crown_flare_deg", "crown_position", "crown_color")},
        "rib_depth": ("ribs > 0", "Only for ribbed or lobed fruits"),
        "blush_color": ("blush > 0", "Only with a blush"),
        **{f: ("stripes > 0", "Only with stripes") for f in ("stripe_color", "stripe_count", "stripe_width")},
        "dot_color": ("dots > 0", "Only with lenticels / dots"),
        "stalk_length_cm": ("cluster_berries <= 1", "Bunches use the peduncle and pedicels"),
    },
}


_ROOT = ("organ == 'Taproot'", "Only for storage roots")
_TUBER = ("organ in ('Tubers', 'Tuberous roots')", "Only for tubers and tuberous roots")
_POTATO = ("organ == 'Tubers'", "Only for stem tubers (potato)")
_HEADR = ("organ == 'Head'", "Only for inflorescence heads")
_BULB = ("organ == 'Bulb'", "Only for bulbs and corms")
RULES["Vegetable"] = {
    "profile": {
        **{f: _ROOT for f in ("root_length_cm", "root_diameter_cm", "widest_position", "shoulder", "taper",
                              "exposure", "tail_cm", "rings", "rootlets", "rootlet_ranks", "root_color",
                              "shoulder_color", "shoulder_tint", "tip_color", "tip_tint")},
        **{f: _TUBER for f in ("tuber_count", "tuber_length_cm", "tuber_diameter_cm", "tuber_depth_cm",
                               "tuber_color", "tuber_dots")},
        **{f: _POTATO for f in ("stolon_length_cm", "eyes", "eye_depth")},
        **{f: _BULB for f in ("bulb_diameter_cm", "bulb_shape", "bulb_neck", "cloves", "bulb_exposure",
                              "bulb_color", "bulb_roots", "contractile_roots")},
        **{f: _HEADR for f in ("head_type", "head_diameter_cm", "head_height_ratio", "head_levels", "florets",
                               "floret_scale", "head_color", "branch_color", "head_wrap")},
        "bud_size_mm": ("organ == 'Head' and head_type == 'Buds'", "Only for broccoli-type heads"),
        "stem_radius_mm": ("stem_height_cm > 0 or organ == 'Bulb'", "Only for a visible stem or a bulb neck"),
        "stem_color": ("stem_height_cm > 0", "Only for a visible stem"),
        "rootlet_ranks": ("organ == 'Taproot' and rootlets > 0", "Only with rootlets"),
    },
    "leaf": dict(RULES["Tree"]["leaf_morphology"]),
}

_EAR = ("head == 'Maize' and ears > 0", "Only for maize ears")
RULES["Grass"] = {
    "profile": {
        **{f: ("head != 'None'", "Only when the grass flowers") for f in (
            "head_length_cm", "peduncle_cm", "spikelets", "spikelet_mm", "nod", "head_color")},
        "head_width_cm": ("head in ('Panicle', 'Plume')", "Only for panicles and plumes"),
        "branches": ("head in ('Panicle', 'Plume', 'One-sided', 'Maize')", "Not for spikes"),
        "branch_angle_deg": ("head in ('Panicle', 'Plume', 'One-sided', 'Maize')", "Not for spikes"),
        "awn_cm": ("head in ('Spike', 'Panicle', 'One-sided')", "Only for awned spikelets"),
        "awn_color": ("head != 'None' and (awn_cm > 0 or head == 'Plume')", "Only with awns or plume hairs"),
        "ears": ("head == 'Maize'", "Only for maize"),
        "ligule_mm": ("ligule != 'None'", "Only with a ligule"),
        "auricle_mm": ("auricles != 'None'", "Only with auricles"),
        "root_primordia": ("growth_ring > 0", "Only with a growth ring and root band (sugarcane)"),
        **{f: _EAR for f in ("ear_node", "ear_length_cm", "ear_diameter_cm", "kernel_rows", "husk", "silk_cm",
                             "kernel_color", "silk_color")},
    },
}

_SYM = ("habit == 'Sympodial'", "Only for sympodial orchids (growths along a rhizome)")
_BULB = ("habit == 'Sympodial' and pseudobulb_cm > 0", "Only with pseudobulbs")
_SIDE = ("side_lobes > 0", "Only for 3-lobed lips (side_lobes > 0)")
_CAPS = ("fruit_set > 0", "Only when some flowers set capsules")
RULES["Orchid"] = {
    "profile": {
        **{f: _SYM for f in ("growths", "rhizome_cm", "growth_angle_deg", "leafless_growths", "pseudobulb_cm")},
        **{f: _BULB for f in ("pseudobulb_diameter_cm", "bulb_widest", "bulb_fullness", "bulb_flatten",
                              "bulb_ridges", "sheath_cover")},
        "bulb_lean_deg": _SYM,
        "stem_length_cm": ("habit != 'Sympodial'", "Monopodial stem or climbing vine only"),
        "internode_cm": ("habit != 'Sympodial' or leaf_placement == 'Along'", "Stems and canes with leaves along"),
        **{f: ("habit == 'Climbing'", "Only for climbing orchids (support post)")
           for f in ("support_height_m", "support_radius_cm")},
        "leaf_placement": _SYM,
        "aerial_share": ("habit != 'Climbing'", "Climbing orchids: one root per node"),
        "infl_origin": ("habit == 'Sympodial'", "Monopodial and climbing orchids flower from the leaf axils"),
        "branch_ratio": ("branches > 0", "Only for panicles"),
        **{f: _CAPS for f in ("capsule_cm", "capsule_diameter_cm", "capsule_color")},
        **{f: _SIDE for f in ("side_lobe_pos", "side_lobe_length", "side_lobe_erect_deg", "midlobe_width",
                              "isthmus")},
        "lip_roll_extent": ("lip_roll > 0", "Only for rolled lips"),
        "lip_waves": ("lip_wave > 0", "Only with wavy lip margins"),
        **{f: ("callus_mm > 0", "Only with a callus") for f in ("callus_ridges", "callus_pos")},
        "spot_size_mm": ("spots > 0 or lip_spots > 0", "Only with spots"),
        "spot_stretch": ("spots > 0 or lip_spots > 0", "Only with spots"),
        "tip_color": ("tip_amount > 0", "Only with coloured tips"),
        "mottle_color": ("leaf_mottle > 0", "Only with mottled leaves"),
        "lateral_sepal_deg": ("synsepal < 0.85", "Lateral sepals fused into a synsepal"),
        "mount": ("habit != 'Climbing'", "Climbing orchids grow on their post"),
        "pot_diameter_cm": ("habit != 'Climbing' and mount == 'Pot'", "Only in a pot"),
        "branch_diameter_cm": ("habit != 'Climbing' and mount == 'Branch'", "Only on a branch"),
        "lip_sac_extent": ("lip_sac > 0", "Only for pouched lips"),
        **{f: ("spur_cm > 0", "Only with a spur") for f in ("spur_diameter_mm", "spur_curve")},
    },
}


def condition(form: str, section: str, field: str):
    """(expression, note) of a field, or None when it always applies."""
    return RULES.get(form, {}).get(section, {}).get(field)


def _plain(v):
    return v.value if isinstance(v, Enum) else v


def applies(form: str, section: str, field: str, values) -> bool:
    """Whether `field` applies given the other values of its section (dict or object). Unknown names: True."""
    rule = condition(form, section, field)
    if rule is None:
        return True
    ns = values if isinstance(values, dict) else {k: getattr(values, k) for k in dir(values) if not k.startswith("_")}
    ns = {k: _plain(v) for k, v in ns.items() if not callable(v)}
    try:
        return bool(eval(rule[0], {"__builtins__": {}}, ns))
    except NameError:
        return True


def section_object(form: str, preset, section: str):
    return getattr(preset, section)


def inactive_fields(form: str, preset) -> list[str]:
    """'section.field' of every field that has no effect for this preset."""
    out = []
    for section, rules in RULES.get(form, {}).items():
        obj = section_object(form, preset, section)
        values = {k: getattr(obj, k) for k in vars(obj)}
        for field in rules:
            if not applies(form, section, field, values):
                out.append(f"{section}.{field}")
    return out
