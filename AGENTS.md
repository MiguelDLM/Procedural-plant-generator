# Notes for AI agents

- **Creating plant presets:** read [docs/PRESETS.md](docs/PRESETS.md) (format, workflow, how to turn
  botanical data into traits), then [docs/preset-fields.md](docs/preset-fields.md) (every field with type,
  unit, range, default and meaning). The machine-readable contract is
  [schemas/ppg-preset.schema.json](schemas/ppg-preset.schema.json).
- **Start from a base:** set `"base"` to the closest built-in species of the same growth form and list only
  the differences. `python -m core.presets export-builtin /tmp/builtin` dumps every built-in preset as JSON.
- **Validate before handing a preset over:** `python -m core.presets validate file.json` (no Blender needed)
  must print `OK` with no warnings.
- **Cite sources** in `metadata.sources` and keep values inside the documented ranges.
- **Code:** generation lives in `core/` (pure Python + NumPy, testable without Blender); Blender UI and
  materials in `blender/`. Run the test suite with `python -m unittest discover -s tests`.
- **Regenerating docs:** after changing a trait dataclass or its ranges, run
  `python -m core.presets schema > schemas/ppg-preset.schema.json` and
  `python -m core.presets docs > docs/preset-fields.md`.
