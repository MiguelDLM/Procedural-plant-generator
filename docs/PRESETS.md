# Plant presets (JSON)

A preset describes one plant (tree, cactus, rosette succulent or flower) by **measurable botanical traits**.
Presets are plain JSON files: easy to share, review in git, validate, and write by hand or with an AI agent.

- Schema: [`schemas/ppg-preset.schema.json`](../schemas/ppg-preset.schema.json) (JSON Schema 2020-12).
- Every field, with type, unit, range, default and meaning: [`docs/preset-fields.md`](preset-fields.md).
- Examples: [`presets/examples/`](../presets/examples/).

## In Blender

Panel **Presets (Import / Export)**:

| Button | What it does |
|---|---|
| **Save as New Preset** | Saves the current sliders as a new preset in your personal library. It appears at once in the species menu, marked `[user]`. |
| **Export** | Writes the current plant to a `.json` file to share. *Only Changes* keeps just the values that differ from the base preset. |
| **Import** | Validates one or more `.json` files, copies them into your library and selects the last one. |
| **Import from URL** | Downloads a shared preset (an `https://` raw JSON link, e.g. a GitHub raw file or a gist). Needs *Allow Online Access* in Blender's preferences. |
| **Folder / Reload** | Opens your library folder; reloads it after you add files by hand. |
| **Remove User Preset** | Deletes the selected user preset (built-in presets cannot be removed). |

User presets live in the add-on's user folder (Blender 4.2+ extensions) or, for legacy installs, in
`<Blender config>/procedural_plant_generator/presets/`. Every `.json` there is loaded when the add-on starts.
A user preset with the same `id` as a built-in one replaces it.

## File format

```json
{
  "$schema": "https://raw.githubusercontent.com/MiguelDLM/Procedural-plant-generator/main/schemas/ppg-preset.schema.json",
  "format": "ppg-preset",
  "format_version": 1,
  "growth_form": "Tree",
  "id": "quercus_ilex",
  "base": "quercus_agrifolia",
  "default_flower": "catkin_amentum",
  "metadata": {
    "author": "Your name",
    "license": "CC-BY-4.0",
    "description": "Holm oak: evergreen Mediterranean oak ...",
    "sources": ["Vaucher H. (2003) Tree Bark: A Color Guide"],
    "tags": ["Mediterranean", "evergreen"]
  },
  "preset": {
    "scientific_name": "Quercus ilex",
    "allometry": {"height_max_m": 22.0},
    "leaf_morphology": {"blade_length_cm": 4.5, "adaxial_color": [0.10, 0.20, 0.08]},
    "bark": {"pattern": "Fissured", "blockiness": 0.8}
  }
}
```

| Key | Required | Meaning |
|---|---|---|
| `format`, `format_version` | yes | Always `"ppg-preset"` and `1`. |
| `growth_form` | yes | `Tree`, `Cactus`, `Rosette` or `Flower`. |
| `id` | yes | Unique key, `lower_snake_case` (letter first; letters, digits, `_`; 2–64 chars). Usually the binomial: `agave_salmiana`. |
| `base` | no | Id of a built-in or library preset **of the same growth form** to start from. With a base, `preset` lists only what differs. |
| `default_flower` | no | Flower preset shown on the plant (`Tree`, `Cactus`, `Rosette`). |
| `metadata` | no | Author, licence, description, sources, tags. Always cite your sources. |
| `preset` | yes | The trait values, nested exactly like the field reference. |

### What `preset` contains per growth form

- **Tree** → `scientific_name`, `common_name`, `family`, `biome`, `growth_habit`, `notes`, `deciduous`, and the
  sections `allometry`, `architecture`, `leaf_morphology`, `venation`, `biomechanics`, `bark`, `roots`.
- **Cactus / Rosette** → `scientific_name`, `common_name`, `family`, `biome`, `notes`, `form`, and `profile`
  (cactus or rosette traits).
- **Flower** → `scientific_name`, `common_name`, `family`, `formula` (floral formula), `notes`, and `flower`
  (floral organs) and `infl` (inflorescence).

### Conventions

- Units are in the field name: `_m` metres, `_cm` centimetres, `_mm` millimetres, `_deg` degrees,
  `_g_cm3` g/cm³. Ratios and amounts without a suffix are dimensionless (often 0–1).
- Colours are sRGB triplets with components 0–1: `[0.10, 0.20, 0.08]`. Values given as 0–255 are rescaled,
  with a warning.
- Enumerations are written as their value (`"Fissured"`, `"Taproot"`, `"Columnar"`); case does not matter.
- Values outside the documented range are clamped; unknown fields are ignored. Both produce warnings.

## Validating (no Blender needed)

```bash
python -m core.presets validate my_plant.json other.json
```

It prints `OK` or `ERROR` for each file and lists every warning: unknown field, value clamped, wrong type,
missing base. Fix the warnings until none remain. Other commands:

```bash
python -m core.presets schema > schemas/ppg-preset.schema.json   # regenerate the schema
python -m core.presets docs   > docs/preset-fields.md            # regenerate the field reference
python -m core.presets export-builtin presets/builtin            # every built-in preset as JSON (examples)
```

## Creating a preset for a new species (guide for people and AI agents)

1. **Pick the growth form and a base.** Choose the closest built-in species of the same form (same
   family or similar habit). Start with `export-builtin` and read the base's JSON to see every value in
   context. Writing only differences on top of a good base is safer than filling every field.
2. **Collect measurable data, and cite it** in `metadata.sources`: floras and monographs, species
   databases, allometry papers, bark guides. Prefer measured ranges and use mid values.
3. **Translate data into traits** (see [`preset-fields.md`](preset-fields.md) for every field):

   | Data you find | Fields |
   |---|---|
   | Mature height, trunk diameter (DBH at 1.3 m) | `allometry.height_max_m`, `allometry.dbh_default_m`; adjust `height_a`/`height_b` only if you have an allometric fit H = a·DBH_cm^b |
   | Crown width, crown shape (conical, rounded, vase) | `allometry.crown_radius_c`/`_d`; `architecture.crown_widest_position`, `crown_fullness` |
   | Branching habit, branch angles, opposite/alternate leaves | `architecture.model` (Hallé–Oldeman), `branch_angle_mean_deg`, `phyllotaxis` |
   | Leaf length, shape, margin, lobes, compound leaves | `leaf_morphology.*` (Ellis et al. 2009 vocabulary) |
   | Leaf colours (upper, lower, autumn) | `adaxial_color`, `abaxial_color`, `autumn_color` |
   | Bark description ("deeply furrowed", "flaking plates", "smooth grey") | `bark.pattern`, `blockiness`, `segments`, `plate_tilt`, `warp`, colours, `onset_radius_cm` |
   | Rooting (taproot, shallow plate, buttresses), rooting depth | `roots.system`, `max_depth_m`, `lateral_count` |
   | Cactus height, stem diameter, rib count, spines per areole and their length | `profile.*` of a Cactus |
   | Rosette diameter, leaf length and number, teeth, terminal spine | `profile.*` of a Rosette |
   | Floral formula, petal number and size, colour, inflorescence type | `flower.*`, `infl.*` |

4. **Keep values inside the documented ranges** and consistent with each other. For example,
   `elevation_inner_deg` should be greater than `elevation_outer_deg`, and `arm_height_min` lower than
   `arm_height_max`.
5. **Validate** with `python -m core.presets validate`, then import the file in Blender, generate the plant
   and compare it with photographs. Iterate on the few traits that matter visually.

### Minimal examples

Tree, partial (base + differences):

```json
{"format": "ppg-preset", "format_version": 1, "growth_form": "Tree", "id": "mossy_oak",
 "base": "quercus_robur", "preset": {"bark": {"moss": 0.8}}}
```

Cactus, partial:

```json
{"format": "ppg-preset", "format_version": 1, "growth_form": "Cactus", "id": "stenocereus_thurberi",
 "base": "pachycereus_marginatus",
 "preset": {"scientific_name": "Stenocereus thurberi",
            "profile": {"height_m": 5.0, "rib_count": 15, "offsets": 12, "spine_color": [0.35, 0.25, 0.20]}}}
```

The full files in [`presets/examples/`](../presets/examples/) show a tree (*Quercus ilex*), a cactus
(*Stenocereus thurberi*), a rosette (*Agave salmiana*) and a flower (*Zinnia elegans*).

## Sharing

- Share the `.json` files directly, publish them in a repository, or put them in a gist and share the raw
  link (*Import from URL*).
- Choose a licence in `metadata.license` (`CC-BY-4.0` by default; `CC0-1.0` for public domain).
- Built-in ids are reserved for the add-on's own presets; use the species binomial plus a suffix for
  variants (`quercus_robur_pollard`).
- The format is versioned (`format_version`). Newer versions of the add-on read older files; a file
  with a newer version than the add-on supports is rejected with a clear message.
