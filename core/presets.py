"""
Plant presets as JSON: serialisation, validation, JSON Schema and field reference.

A preset file ("PPG preset", format version 1) is a small envelope around the trait tree of one plant:

    {
      "$schema": "https://raw.githubusercontent.com/MiguelDLM/Procedural-plant-generator/main/schemas/ppg-preset.schema.json",
      "format": "ppg-preset",
      "format_version": 1,
      "growth_form": "Tree" | "Cactus" | "Rosette" | "Flower",
      "id": "quercus_robur_old_growth",          # lower_snake_case, unique in the library
      "base": "quercus_robur",                   # optional: built-in or library preset to start from
      "default_flower": "rosa_canina",           # optional (Tree / Cactus / Rosette)
      "metadata": {"author": "", "license": "CC-BY-4.0", "description": "", "sources": [], "tags": []},
      "preset": { ... trait values, nested like the dataclasses ... }
    }

With "base", "preset" only needs the values that differ (partial preset); without it, missing values take
the class defaults. Unknown keys and out-of-range numbers are reported (numbers are clamped).

Command line (no Blender needed):
    python -m core.presets validate my_plant.json [...]
    python -m core.presets schema  > schemas/ppg-preset.schema.json
    python -m core.presets docs    > docs/preset-fields.md
    python -m core.presets export-builtin presets/builtin
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import inspect
import json
import math
import re
import sys
from enum import Enum
from typing import Any

FORMAT = "ppg-preset"
FORMAT_VERSION = 1
FORMS = ("Tree", "Cactus", "Rosette", "Flower")
SCHEMA_URL = ("https://raw.githubusercontent.com/MiguelDLM/Procedural-plant-generator/main/"
              "schemas/ppg-preset.schema.json")
ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
ENUM_BITS = 23        # Menu numbers must be exact in a float32 (Blender stores enum values with 24-bit precision)


def item_number(key: str) -> int:
    """Stable menu number for a preset id (saved .blend files keep pointing at the same preset). Kept below
    2**23: larger numbers lost their low bits when Blender stored the selection, so the menu no longer found
    the chosen preset (e.g. Opuntia 779024583 came back as 779024576)."""
    import zlib
    return zlib.crc32(key.encode("utf-8")) & ((1 << ENUM_BITS) - 1)


def enum_numbers(keys) -> dict:
    """Unique numbers for a list of keys (collisions resolved deterministically in sorted key order)."""
    used, out = set(), {}
    for k in sorted(keys):
        n = item_number(k)
        while n in used:
            n = (n + 1) & ((1 << ENUM_BITS) - 1)
        used.add(n)
        out[k] = n
    return out


# -----------------------------------------------------------------------------
# Catalogues and ranges (imported lazily to keep this module light)
# -----------------------------------------------------------------------------
def _catalog(form: str) -> dict:
    if form == "Tree":
        from .species_db import SPECIES_CATALOG
        return SPECIES_CATALOG
    if form in ("Cactus", "Rosette"):
        from .succulent_db import CATALOGS, GrowthForm
        return CATALOGS[GrowthForm(form)]
    if form == "Flower":
        from .flower_db import FLOWER_CATALOG
        return FLOWER_CATALOG
    raise ValueError(f"Unknown growth form {form!r}; expected one of {FORMS}")


def _flower_map(form: str) -> dict | None:
    from .flower_db import TREE_FLOWERS, CACTUS_FLOWERS, ROSETTE_FLOWERS
    return {"Tree": TREE_FLOWERS, "Cactus": CACTUS_FLOWERS, "Rosette": ROSETTE_FLOWERS}.get(form)


def _template(form: str):
    """A preset object with every field at its class default (used when no base is given)."""
    if form == "Tree":
        from .species_preset import BotanicalSpeciesPreset
        from .allometry import AllometricProfile
        from .architecture import ArchitectureProfile
        from .leaf_morphology import LeafMorphologyProfile
        from .leaf_venation import VenationProfile
        from .biomechanics import BiomechanicalProfile
        return BotanicalSpeciesPreset("", "", "", "", "", AllometricProfile(), ArchitectureProfile(),
                                      LeafMorphologyProfile(), VenationProfile(), BiomechanicalProfile())
    if form in ("Cactus", "Rosette"):
        from .succulent_db import SucculentPreset, PROFILE_CLASSES, GrowthForm
        gf = GrowthForm(form)
        return SucculentPreset("", "", "", "", gf, PROFILE_CLASSES[gf]())
    if form == "Flower":
        from .flower_db import FlowerPreset
        from .flower import FlowerProfile
        from .inflorescence import InflorescenceProfile
        return FlowerPreset("", "", "", "", "", FlowerProfile(), InflorescenceProfile())
    raise ValueError(form)


def _ranges(form: str) -> dict:
    """Allowed numeric ranges by dotted path inside "preset"."""
    out = {}
    if form == "Tree":
        from .trait_space import TRAITS
        for t in TRAITS:
            out[t.path] = (t.lo, t.hi)
    elif form in ("Cactus", "Rosette"):
        from .succulent_db import RANGES, GrowthForm
        out = {f"profile.{k}": v for k, v in RANGES[GrowthForm(form)].items()}
    elif form == "Flower":
        from .flower_db import FLOWER_RANGES, INFL_RANGES
        out = {f"flower.{k}": v for k, v in FLOWER_RANGES.items()}
        out.update({f"infl.{k}": v for k, v in INFL_RANGES.items()})
    return out


# -----------------------------------------------------------------------------
# Serialisation
# -----------------------------------------------------------------------------
def to_plain(obj: Any) -> Any:
    """Dataclasses -> dicts, Enums -> values, tuples -> lists, NumPy scalars -> Python numbers."""
    if dataclasses.is_dataclass(obj):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _diff(a: Any, b: Any) -> Any:
    """Values of `a` that differ from `b` (nested dicts); None when equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        for k, v in a.items():
            d = _diff(v, b.get(k, None)) if k in b else v
            if d is not None:
                out[k] = d
        return out or None
    if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return None if math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9) else a
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return None if all(_diff(x, y) is None for x, y in zip(a, b)) else a
    return None if a == b else a


def export_preset(form: str, preset_id: str, obj, base: str | None = None, diff: bool = False,
                  metadata: dict | None = None, default_flower: str | None = None) -> dict:
    """Envelope for `obj` (a preset object of `form`). With diff=True and a base, only differences are kept."""
    body = to_plain(obj)
    if diff and base:
        base_obj = _catalog(form).get(base)
        if base_obj is None:
            raise KeyError(f"Base preset {base!r} not found for {form}")
        body = _diff(body, to_plain(base_obj)) or {}
    meta = {"author": "", "license": "CC-BY-4.0", "description": "", "sources": [], "tags": [],
            "created": datetime.date.today().isoformat(), "generator_version": _generator_version()}
    meta.update(metadata or {})
    env = {"$schema": SCHEMA_URL, "format": FORMAT, "format_version": FORMAT_VERSION, "growth_form": form,
           "id": preset_id}
    if base:
        env["base"] = base
    if default_flower:
        env["default_flower"] = default_flower
    env["metadata"] = meta
    env["preset"] = body
    return env


def _generator_version() -> str:
    try:
        import os
        import tomllib
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "blender_manifest.toml"), "rb") as fh:
            return tomllib.load(fh).get("version", "")
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Loading and validation
# -----------------------------------------------------------------------------
class PresetError(ValueError):
    pass


def _coerce(template: Any, value: Any, path: str, ranges: dict, warn: list) -> Any:
    """Value converted to the type of `template` (recursively for dataclasses), clamped to `ranges`."""
    if dataclasses.is_dataclass(template):
        if not isinstance(value, dict):
            warn.append(f"{path}: expected an object, got {type(value).__name__}; kept default")
            return template
        obj = copy.deepcopy(template)
        ftypes = {f.name: f.type for f in dataclasses.fields(obj)}
        for k, v in value.items():
            if k not in ftypes:
                warn.append(f"{path}.{k}: unknown field ignored" if path else f"{k}: unknown field ignored")
                continue
            cur = getattr(obj, k)
            # Use the declared type: presets written with literal ints (e.g. 5) still hold float traits
            if ftypes[k] in (float, "float") and isinstance(cur, int) and not isinstance(cur, bool):
                cur = float(cur)
            setattr(obj, k, _coerce(cur, v, f"{path}.{k}" if path else k, ranges, warn))
        return obj
    if isinstance(template, Enum):
        cls = type(template)
        for e in cls:
            if value == e.value or (isinstance(value, str) and value.lower() in (e.value.lower(), e.name.lower())):
                return e
        warn.append(f"{path}: {value!r} is not one of {[e.value for e in cls]}; kept {template.value!r}")
        return template
    if isinstance(template, bool):
        if isinstance(value, bool):
            return value
        warn.append(f"{path}: expected true/false, got {value!r}; kept {template}")
        return template
    if isinstance(template, (int, float)) and not isinstance(template, bool):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            warn.append(f"{path}: expected a number, got {value!r}; kept {template}")
            return template
        v = value
        if path in ranges:
            lo, hi = ranges[path]
            if v < lo or v > hi:
                warn.append(f"{path}: {value} outside [{lo}, {hi}]; clamped")
                v = min(max(v, lo), hi)
        return int(round(v)) if isinstance(template, int) else float(v)
    if isinstance(template, tuple) or (template is None and isinstance(value, list)):
        n = len(template) if isinstance(template, tuple) else len(value)
        if (not isinstance(value, (list, tuple)) or len(value) != n
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)):
            warn.append(f"{path}: expected {n} numbers, got {value!r}; kept default")
            return template
        if n == 3 and all(0.0 <= float(x) <= 1.0 for x in (template or value)) and any(x > 1 for x in value):
            warn.append(f"{path}: colour components must be 0..1 (sRGB); scaled from 0..255")
            value = [x / 255.0 for x in value]
        return tuple(float(x) for x in value)
    if isinstance(template, str):
        if not isinstance(value, str):
            warn.append(f"{path}: expected text, got {value!r}; kept default")
            return template
        return value
    return value


def load_preset(data: dict) -> tuple[str, str, Any, list[str], dict]:
    """
    Validates and builds a preset from a parsed JSON envelope.
    Returns (growth_form, id, preset_object, warnings, envelope_extras). Raises PresetError on fatal errors.
    """
    if not isinstance(data, dict):
        raise PresetError("A preset file must contain a JSON object")
    if data.get("format") != FORMAT:
        raise PresetError(f"'format' must be {FORMAT!r}")
    ver = data.get("format_version")
    if not isinstance(ver, int) or ver > FORMAT_VERSION:
        raise PresetError(f"Unsupported format_version {ver!r} (this version reads <= {FORMAT_VERSION})")
    form = data.get("growth_form")
    if form not in FORMS:
        raise PresetError(f"'growth_form' must be one of {FORMS}")
    pid = data.get("id")
    if not isinstance(pid, str) or not ID_RE.match(pid):
        raise PresetError("'id' must be lower_snake_case: a letter, then letters, digits or _ (2-64 chars)")
    warn: list[str] = []
    base = data.get("base")
    if base is not None:
        base_obj = _catalog(form).get(base)
        if base_obj is None:
            raise PresetError(f"Base preset {base!r} not found among {form} presets")
        template = copy.deepcopy(base_obj)
    else:
        template = _template(form)
    body = data.get("preset", {})
    if not isinstance(body, dict):
        raise PresetError("'preset' must be an object")
    known = {"$schema", "format", "format_version", "growth_form", "id", "base", "default_flower", "metadata",
             "preset"}
    for k in data:
        if k not in known:
            warn.append(f"{k}: unknown top-level key ignored")
    obj = _coerce(template, body, "", _ranges(form), warn)
    if not getattr(obj, "scientific_name", ""):
        warn.append("scientific_name is empty")
    if base is None:
        missing = _count_missing(template, body)
        if missing:
            warn.append(f"{missing} values not given and no 'base': class defaults were used "
                        "(prefer 'base' with a related species)")
    flower = data.get("default_flower")
    if flower is not None:
        from .flower_db import FLOWER_CATALOG
        if form == "Flower":
            warn.append("default_flower is ignored for Flower presets")
        elif flower not in FLOWER_CATALOG:
            warn.append(f"default_flower {flower!r} not found; ignored")
            flower = None
    extras = {"metadata": data.get("metadata", {}), "default_flower": flower, "base": base}
    return form, pid, obj, warn, extras


def _count_missing(template, body) -> int:
    if dataclasses.is_dataclass(template):
        n = 0
        for f in dataclasses.fields(template):
            sub = body.get(f.name) if isinstance(body, dict) else None
            if sub is None:
                n += _leaf_count(getattr(template, f.name))
            else:
                n += _count_missing(getattr(template, f.name), sub)
        return n
    return 0


def _leaf_count(v) -> int:
    return sum(_leaf_count(getattr(v, f.name)) for f in dataclasses.fields(v)) if dataclasses.is_dataclass(v) else 1


def register_preset(form: str, pid: str, obj, default_flower: str | None = None) -> bool:
    """Adds (or replaces) a preset in the running catalogue. Returns True if it replaced one."""
    cat = _catalog(form)
    replaced = pid in cat
    cat[pid] = obj
    fm = _flower_map(form)
    if fm is not None and default_flower:
        fm[pid] = default_flower
    return replaced


def load_file(path: str, register: bool = True) -> tuple[str, str, list[str]]:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as e:
            raise PresetError(f"Invalid JSON: {e}") from e
    form, pid, obj, warn, extras = load_preset(data)
    if register:
        register_preset(form, pid, obj, extras.get("default_flower"))
    return form, pid, warn


def save_file(path: str, envelope: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# -----------------------------------------------------------------------------
# Field documentation (descriptions, units) for the schema and the reference
# -----------------------------------------------------------------------------
def _source_comments(cls) -> dict:
    """Inline (# ...) or immediately preceding comment of each field in a dataclass's source."""
    try:
        lines = inspect.getsource(cls).splitlines()
    except (OSError, TypeError):
        return {}
    out, pending = {}, []
    for ln in lines:
        s = ln.strip()
        if s.startswith("#"):
            pending.append(s.lstrip("# ").strip())
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*:\s*[^=]+(=.*)?$", s)
        if m and not s.startswith(("def ", "class ", "@", "return")):
            inline = s.split("#", 1)[1].strip() if "#" in s else ""
            out[m.group(1)] = inline or " ".join(pending)
        if not s.startswith("#"):
            pending = [] if s else pending
    return out


def _unit(name: str) -> str:
    for suf, u in (("_deg_per_m", "deg/m"), ("_g_cm3", "g/cm^3"), ("_mm_per_mm2", "mm/mm^2"), ("_deg", "deg"),
                   ("_cm", "cm"), ("_mm", "mm"), ("_m", "m"), ("_pct", "%")):
        if name.endswith(suf):
            return u
    return ""


def describe(cls, name: str) -> str:
    from .preset_docs import FIELD_DOCS
    key = f"{cls.__name__}.{name}"
    return FIELD_DOCS.get(key) or _source_comments(cls).get(name, "")


def _json_type(v) -> dict:
    if isinstance(v, bool):
        return {"type": "boolean"}
    if isinstance(v, int):
        return {"type": "integer"}
    if isinstance(v, float):
        return {"type": "number"}
    if isinstance(v, str):
        return {"type": "string"}
    if isinstance(v, tuple):
        if len(v) == 3:
            return {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 3,
                    "maxItems": 3, "x-kind": "colour (sRGB 0..1)"}
        return {"type": "array", "items": {"type": "number"}, "minItems": len(v), "maxItems": len(v)}
    if v is None:
        return {"type": ["array", "null"], "items": {"type": "number"}}
    return {}


def _class_schema(cls, instance, prefix: str, ranges: dict, defs: dict, form: str = "",
                  section: str = "") -> dict:
    from .relevance import condition
    props = {}
    for f in dataclasses.fields(cls):
        v = getattr(instance, f.name)
        path = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(v):
            sub = type(v).__name__
            if sub not in defs:
                defs[sub] = None
                defs[sub] = _class_schema(type(v), v, f"{path}.", ranges, defs, form, f.name)
            props[f.name] = {"$ref": f"#/$defs/{sub}"}
            continue
        if isinstance(v, Enum):
            p = {"type": "string", "enum": [e.value for e in type(v)]}
        else:
            p = _json_type(v)
        desc = describe(cls, f.name)
        if desc:
            p["description"] = desc
        u = _unit(f.name)
        if u:
            p["x-unit"] = u
        if path in ranges:
            p["minimum"], p["maximum"] = ranges[path]
        dv = to_plain(v)
        if dv is not None:
            p["default"] = dv
        rule = condition(form, section, f.name)
        if rule is not None:
            p["x-applies-when"] = rule[0]
            p["description"] = (p.get("description", "") + f" Applies when: {rule[0]}.").strip()
        props[f.name] = p
    return {"type": "object", "additionalProperties": False, "properties": props,
            "description": (inspect.getdoc(cls) or "").split("\n\n")[0]}


def json_schema() -> dict:
    defs: dict = {}
    forms = {}
    for form in FORMS:
        t = _template(form)
        name = {"Tree": "TreePreset", "Cactus": "CactusPreset", "Rosette": "RosettePreset",
                "Flower": "FlowerPreset"}[form]
        sch = _class_schema(type(t), t, "", _ranges(form), defs, form, "")
        sch["description"] = f"Trait values of a {form} preset (all optional when 'base' is given)."
        defs[name] = sch
        forms[form] = name
    # Profile classes may be shared by name between forms only if identical; keep first definition
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_URL,
        "title": "Procedural Plant Generator preset",
        "description": "One plant (tree, cactus, rosette succulent or flower) described by measurable traits. "
                       "See docs/PRESETS.md.",
        "type": "object",
        "required": ["format", "format_version", "growth_form", "id", "preset"],
        "properties": {
            "$schema": {"type": "string"},
            "format": {"const": FORMAT},
            "format_version": {"type": "integer", "minimum": 1, "maximum": FORMAT_VERSION},
            "growth_form": {"enum": list(FORMS)},
            "id": {"type": "string", "pattern": ID_RE.pattern,
                   "description": "Unique lower_snake_case identifier (becomes the preset key)."},
            "base": {"type": "string", "description": "Built-in or library preset of the same growth form to "
                                                      "start from; 'preset' then lists only the differences."},
            "default_flower": {"type": "string", "description": "Flower preset id shown on this plant."},
            "metadata": {"type": "object", "properties": {
                "author": {"type": "string"}, "license": {"type": "string"},
                "description": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}}, "created": {"type": "string"},
                "generator_version": {"type": "string"}}},
            "preset": {"type": "object"},
        },
        "allOf": [{"if": {"properties": {"growth_form": {"const": f}}},
                   "then": {"properties": {"preset": {"$ref": f"#/$defs/{n}"}}}} for f, n in forms.items()],
        "$defs": defs,
    }


def fields_reference_md() -> str:
    """Markdown reference of every field (generated from the schema)."""
    sch = json_schema()
    defs = sch["$defs"]
    out = ["# Preset field reference", "",
           "Generated by `python -m core.presets docs` from the code; do not edit by hand.", "",
           "Units: suffix `_m` metres, `_cm` centimetres, `_mm` millimetres, `_deg` degrees. Colours are sRGB "
           "triplets in 0..1. Ranges are the limits enforced on import (values outside are clamped).", ""]
    order = ["TreePreset", "CactusPreset", "RosettePreset", "FlowerPreset"]
    order += [k for k in defs if k not in order]
    for name in order:
        d = defs[name]
        out += [f"## {name}", "", d.get("description", ""), "",
                "| Field | Type | Unit | Range | Default | Meaning |", "|---|---|---|---|---|---|"]
        for f, p in d["properties"].items():
            if "$ref" in p:
                ref = p["$ref"].split("/")[-1]
                out.append(f"| `{f}` | object | | | | see [{ref}](#{ref.lower()}) |")
                continue
            typ = "enum: " + ", ".join(f"`{e}`" for e in p["enum"]) if "enum" in p else (
                p.get("x-kind") or (p.get("type") if isinstance(p.get("type"), str) else "array|null"))
            rng = f"{p['minimum']} – {p['maximum']}" if "minimum" in p else ""
            dv = p.get("default", "")
            dv = json.dumps(dv) if isinstance(dv, (list, bool)) or dv is None else str(dv)
            desc = p.get("description", "").replace("|", "\\|")
            out.append(f"| `{f}` | {typ} | {p.get('x-unit', '')} | {rng} | {dv} | {desc} |")
        out.append("")
    return "\n".join(out) + "\n"


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    import os
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "validate":
        bad = 0
        for p in args:
            try:
                form, pid, warn = load_file(p, register=False)
                print(f"OK    {p}: {form} '{pid}'" + (f" ({len(warn)} warnings)" if warn else ""))
                for w in warn:
                    print(f"      - {w}")
            except (PresetError, OSError) as e:
                bad += 1
                print(f"ERROR {p}: {e}")
        return 1 if bad else 0
    if cmd == "schema":
        print(json.dumps(json_schema(), indent=2, ensure_ascii=False))
        return 0
    if cmd == "docs":
        print(fields_reference_md(), end="")
        return 0
    if cmd == "export-builtin":
        out = args[0] if args else "presets/builtin"
        n = 0
        for form in FORMS:
            os.makedirs(os.path.join(out, form.lower()), exist_ok=True)
            fm = _flower_map(form) or {}
            for key, obj in _catalog(form).items():
                env = export_preset(form, key, obj, default_flower=fm.get(key),
                                    metadata={"author": "Procedural Plant Generator (built-in)",
                                              "description": getattr(obj, "notes", "")})
                save_file(os.path.join(out, form.lower(), f"{key}.json"), env)
                n += 1
        print(f"Exported {n} presets to {out}")
        return 0
    print(f"Unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
