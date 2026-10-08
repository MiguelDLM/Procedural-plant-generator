"""
Blender side of JSON presets: a personal preset library, export / save / import (files or URL), and
species menus that pick up new presets at once (dynamic enum items with stable ids).
"""

import copy
import json
import os
import shutil
import zlib

try:
    import bpy
    from bpy.types import Operator
    from bpy.props import StringProperty, BoolProperty, CollectionProperty
    from bpy_extras.io_utils import ExportHelper, ImportHelper
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    Operator = ExportHelper = ImportHelper = object

    def StringProperty(**kw):
        return None
    BoolProperty = CollectionProperty = StringProperty

try:
    from ..core import presets as P
    from ..core.species_db import get_preset_names
    from ..core.succulent_db import GrowthForm, preset_items
    from ..core.flower_db import flower_items
except (ImportError, ValueError):
    from core import presets as P
    from core.species_db import get_preset_names
    from core.succulent_db import GrowthForm, preset_items
    from core.flower_db import flower_items

USER_IDS: dict = {form: set() for form in P.FORMS}     # Presets loaded from the personal library
_ITEMS: dict = {}                                       # Keeps enum item strings alive (Blender requirement)
MAX_URL_BYTES = 2_000_000


# -----------------------------------------------------------------------------
# Dynamic menus
# -----------------------------------------------------------------------------
item_number = P.item_number
_numbers = P.enum_numbers


def _base_items(form: str) -> list:
    if form == "Tree":
        return get_preset_names()
    if form == "Flower":
        return flower_items()
    return preset_items(GrowthForm(form))


def enum_items(form: str):
    def items(self, context):
        out = []
        base = _base_items(form)
        nums = _numbers([k for k, _, _ in base])
        for key, label, desc in base:
            user = key in USER_IDS[form]
            out.append((key, label + ("  [user]" if user else ""), desc, 'USER' if user else 'NONE', nums[key]))
        _ITEMS[form] = out
        return out
    return items


# -----------------------------------------------------------------------------
# Library
# -----------------------------------------------------------------------------
def library_dir() -> str:
    """Personal preset folder (extension user directory, or Blender's config folder for legacy installs)."""
    path = os.environ.get("PPG_PRESET_DIR")      # Override (tests, studio-wide shared libraries)
    if path:
        os.makedirs(path, exist_ok=True)
        return path
    pkg = (__package__ or "").rpartition(".")[0]
    try:
        if pkg:
            path = bpy.utils.extension_path_user(pkg, path="presets", create=True)
    except Exception:
        path = None
    if not path:
        path = bpy.utils.user_resource('CONFIG', path="procedural_plant_generator/presets", create=True)
    os.makedirs(path, exist_ok=True)
    return path


def load_library(report=print) -> int:
    """Registers every preset file of the personal library; returns how many loaded."""
    n = 0
    d = library_dir()
    for name in sorted(os.listdir(d)):
        if not name.lower().endswith(".json"):
            continue
        try:
            form, pid, warn = P.load_file(os.path.join(d, name))
            USER_IDS[form].add(pid)
            n += 1
            for w in warn:
                report(f"[PPG preset {name}] {w}")
        except (P.PresetError, OSError) as e:
            report(f"[PPG preset {name}] not loaded: {e}")
    return n


def _species_prop(form: str) -> str:
    return {"Tree": "species_enum", "Cactus": "cactus_species", "Rosette": "rosette_species",
            "Flower": "flower_species"}[form]


def current_preset(props):
    """(form, base_key, preset_object, default_flower) from the current sliders."""
    form = props.growth_form
    if form == "Tree":
        from .runtime import build_custom_preset_from_props
        obj, _ = build_custom_preset_from_props(props)
        return form, props.species_enum, obj, (props.flower_species if props.show_flowers else None)
    if form in ("Cactus", "Rosette"):
        from .succulents import profile_from_props
        from ..core.succulent_db import CATALOGS
        gf = GrowthForm(form)
        key = props.cactus_species if gf == GrowthForm.CACTUS else props.rosette_species
        if key not in CATALOGS[gf]:
            raise P.PresetError(f"No valid {form} preset selected")
        obj = copy.deepcopy(CATALOGS[gf][key])
        obj.profile = profile_from_props(props, gf, key)
        return form, key, obj, (props.flower_species if props.show_flowers else None)
    from .flowers import flower_from_props
    from ..core.flower_db import FLOWER_CATALOG
    if props.flower_species not in FLOWER_CATALOG:
        raise P.PresetError("No valid flower preset selected")
    obj = copy.deepcopy(FLOWER_CATALOG[props.flower_species])
    obj.flower, obj.infl = flower_from_props(props)
    return form, props.flower_species, obj, None


def select_preset(context, form: str, pid: str):
    """Shows a preset in the panel (switching growth form if needed); its traits load into the sliders."""
    props = context.scene.ppg_properties
    if props.growth_form != form:
        props.growth_form = form
    setattr(props, _species_prop(form), pid)


def _identity(obj, scientific: str, common: str, family: str):
    if scientific:
        obj.scientific_name = scientific
    if common:
        obj.common_name = common
    if family:
        obj.family = family


def _metadata(op) -> dict:
    return {"author": op.author, "license": op.license, "description": op.description,
            "sources": [s.strip() for s in op.sources.split(";") if s.strip()],
            "tags": [t.strip() for t in op.tags.split(",") if t.strip()]}


# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------
class _MetaProps:
    author: StringProperty(name="Author", default="")
    license: StringProperty(name="License", default="CC-BY-4.0",
                            description="Licence for sharing, e.g. CC-BY-4.0, CC0-1.0")
    description: StringProperty(name="Description", default="")
    sources: StringProperty(name="Sources", default="",
                            description="Literature or data sources, separated by ';'")
    tags: StringProperty(name="Tags", default="", description="Comma-separated keywords")


class PPG_OT_PresetExport(Operator, ExportHelper, _MetaProps):
    """Export the current plant (all sliders) as a shareable JSON preset"""
    bl_idname = "ppg.preset_export"
    bl_label = "Export Preset"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    preset_id: StringProperty(name="Preset Id", default="my_plant",
                              description="lower_snake_case identifier of the preset")
    only_changes: BoolProperty(name="Only Changes", default=False,
                               description="Store only the values that differ from the base preset "
                                           "(small file that needs the base to load)")

    def invoke(self, context, event):
        try:
            form, base, obj, _ = current_preset(context.scene.ppg_properties)
        except P.PresetError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.preset_id = f"{base}_custom"
        self.filepath = bpy.path.ensure_ext(os.path.join(os.path.dirname(self.filepath or ""), self.preset_id),
                                            ".json")
        return super().invoke(context, event)

    def execute(self, context):
        try:
            form, base, obj, flower = current_preset(context.scene.ppg_properties)
        except P.PresetError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if not P.ID_RE.match(self.preset_id):
            self.report({'ERROR'}, "Preset id must be lower_snake_case (letters, digits, _)")
            return {'CANCELLED'}
        env = P.export_preset(form, self.preset_id, obj, base=base, diff=self.only_changes,
                              metadata=_metadata(self), default_flower=flower)
        P.save_file(self.filepath, env)
        self.report({'INFO'}, f"Preset saved to {self.filepath}")
        return {'FINISHED'}


class PPG_OT_PresetSave(Operator, _MetaProps):
    """Save the current sliders as a new preset in your personal library (appears in the species menu)"""
    bl_idname = "ppg.preset_save"
    bl_label = "Save as New Preset"
    bl_options = {'REGISTER'}
    preset_id: StringProperty(name="Preset Id", default="my_plant")
    scientific_name: StringProperty(name="Scientific Name", default="")
    common_name: StringProperty(name="Common Name", default="")
    family: StringProperty(name="Family", default="")

    def invoke(self, context, event):
        try:
            form, base, obj, _ = current_preset(context.scene.ppg_properties)
        except P.PresetError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.preset_id = f"{base}_custom"
        self.scientific_name = obj.scientific_name
        self.common_name = obj.common_name
        self.family = obj.family
        return context.window_manager.invoke_props_dialog(self, width=420)

    def execute(self, context):
        if not P.ID_RE.match(self.preset_id):
            self.report({'ERROR'}, "Preset id must be lower_snake_case (letters, digits, _)")
            return {'CANCELLED'}
        try:
            form, base, obj, flower = current_preset(context.scene.ppg_properties)
        except P.PresetError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        _identity(obj, self.scientific_name, self.common_name, self.family)
        env = P.export_preset(form, self.preset_id, obj, base=base if base != self.preset_id else None,
                              diff=False, metadata=_metadata(self), default_flower=flower)
        path = os.path.join(library_dir(), f"{self.preset_id}.json")
        P.save_file(path, env)
        P.register_preset(form, self.preset_id, obj, flower)
        USER_IDS[form].add(self.preset_id)
        select_preset(context, form, self.preset_id)
        self.report({'INFO'}, f"Saved '{self.preset_id}' to your preset library")
        return {'FINISHED'}


def _install(context, data: dict, source: str, op) -> str | None:
    """Validates, copies into the library and registers one preset envelope; returns its id."""
    form, pid, obj, warn, extras = P.load_preset(data)
    P.save_file(os.path.join(library_dir(), f"{pid}.json"), data)
    P.register_preset(form, pid, obj, extras.get("default_flower"))
    USER_IDS[form].add(pid)
    for w in warn:
        print(f"[PPG preset {source}] {w}")
    op.report({'WARNING'} if warn else {'INFO'},
              f"Imported {form} preset '{pid}'" + (f" with {len(warn)} warnings (see console)" if warn else ""))
    return form, pid


class PPG_OT_PresetImport(Operator, ImportHelper):
    """Import JSON presets into your library; they appear in the species menus"""
    bl_idname = "ppg.preset_import"
    bl_label = "Import Presets"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement if BLENDER_AVAILABLE else None,
                              options={'HIDDEN', 'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN'})

    def execute(self, context):
        paths = [os.path.join(self.directory, f.name) for f in self.files if f.name] or [self.filepath]
        last = None
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    last = _install(context, json.load(fh), os.path.basename(path), self)
            except (P.PresetError, OSError, json.JSONDecodeError) as e:
                self.report({'ERROR'}, f"{os.path.basename(path)}: {e}")
        if last:
            select_preset(context, *last)
        return {'FINISHED'} if last else {'CANCELLED'}


class PPG_OT_PresetImportURL(Operator):
    """Download a shared preset (raw JSON link, e.g. a GitHub raw file or gist) into your library"""
    bl_idname = "ppg.preset_import_url"
    bl_label = "Import from URL"
    url: StringProperty(name="URL", default="")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=520)

    def execute(self, context):
        if not getattr(bpy.app, "online_access", True):
            self.report({'ERROR'}, "Online access is disabled (Preferences > System > Network)")
            return {'CANCELLED'}
        if not self.url.lower().startswith("https://"):
            self.report({'ERROR'}, "Only https:// links are accepted")
            return {'CANCELLED'}
        import urllib.request
        try:
            with urllib.request.urlopen(self.url, timeout=15) as r:
                raw = r.read(MAX_URL_BYTES + 1)
            if len(raw) > MAX_URL_BYTES:
                raise P.PresetError("File too large for a preset")
            res = _install(context, json.loads(raw.decode("utf-8")), self.url, self)
        except (P.PresetError, OSError, ValueError) as e:
            self.report({'ERROR'}, f"Could not import: {e}")
            return {'CANCELLED'}
        select_preset(context, *res)
        return {'FINISHED'}


class PPG_OT_PresetOpenFolder(Operator):
    """Open your personal preset folder (share these files, or drop others' presets here)"""
    bl_idname = "ppg.preset_open_folder"
    bl_label = "Open Preset Folder"

    def execute(self, context):
        bpy.ops.wm.path_open(filepath=library_dir())
        return {'FINISHED'}


class PPG_OT_PresetReload(Operator):
    """Reload every preset of your personal folder"""
    bl_idname = "ppg.preset_reload"
    bl_label = "Reload Library"

    def execute(self, context):
        n = load_library()
        self.report({'INFO'}, f"{n} presets loaded from {library_dir()}")
        return {'FINISHED'}


class PPG_OT_PresetRemove(Operator):
    """Delete the selected user preset from your library (built-in presets cannot be removed)"""
    bl_idname = "ppg.preset_remove"
    bl_label = "Remove User Preset"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        props = context.scene.ppg_properties
        form = props.growth_form
        pid = getattr(props, _species_prop(form))
        if pid not in USER_IDS[form]:
            self.report({'WARNING'}, "Only user presets can be removed")
            return {'CANCELLED'}
        path = os.path.join(library_dir(), f"{pid}.json")
        if os.path.exists(path):
            os.remove(path)
        USER_IDS[form].discard(pid)
        P._catalog(form).pop(pid, None)
        self.report({'INFO'}, f"Removed '{pid}' (restart or Reload to restore a built-in of the same id)")
        return {'FINISHED'}


PRESET_CLASSES = (PPG_OT_PresetExport, PPG_OT_PresetSave, PPG_OT_PresetImport, PPG_OT_PresetImportURL,
                  PPG_OT_PresetOpenFolder, PPG_OT_PresetReload, PPG_OT_PresetRemove) if BLENDER_AVAILABLE else ()


def draw_presets(layout, props):
    col = layout.column(align=True)
    col.operator("ppg.preset_save", icon='FILE_NEW')
    row = col.row(align=True)
    row.operator("ppg.preset_export", icon='EXPORT', text="Export")
    row.operator("ppg.preset_import", icon='IMPORT', text="Import")
    col.operator("ppg.preset_import_url", icon='URL')
    row = col.row(align=True)
    row.operator("ppg.preset_open_folder", icon='FILE_FOLDER', text="Folder")
    row.operator("ppg.preset_reload", icon='FILE_REFRESH', text="Reload")
    pid = getattr(props, _species_prop(props.growth_form))
    if pid in USER_IDS.get(props.growth_form, ()):
        col.operator("ppg.preset_remove", icon='TRASH')
