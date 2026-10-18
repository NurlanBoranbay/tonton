"""Settings = {"scene": general thresholds, "rules": {rule_id: {"enabled", **params}}}.

Defaults live in code (scene.DEFAULTS, Rule.params); runs/settings.json overrides them and is what the
next job uses. validate_settings() drops unknown keys and clamps values to each rule's param_schema.
"""
import json, os

from app.rules.base import REGISTRY
from app.rules.scene import DEFAULTS as SCENE_DEFAULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.join(ROOT, "runs", "settings.json")


def defaults():
    from app.rules import violations  # noqa: F401
    return {"scene": dict(SCENE_DEFAULTS),
            "rules": {rid: {"enabled": True, **r.params} for rid, r in REGISTRY.items()}}


def validate_settings(s):
    d = defaults()
    out = {"scene": dict(d["scene"]), "rules": {rid: dict(v) for rid, v in d["rules"].items()}}
    for k, v in (s.get("scene") or {}).items():
        if k in out["scene"] and isinstance(v, (int, float)) and not isinstance(v, bool):
            out["scene"][k] = type(out["scene"][k])(v)
    for rid, cfg in (s.get("rules") or {}).items():
        if rid not in out["rules"]: continue
        schema = REGISTRY[rid].param_schema
        for k, v in (cfg or {}).items():
            if k == "enabled":
                out["rules"][rid]["enabled"] = bool(v)
            elif k in out["rules"][rid] and isinstance(v, (int, float)) and not isinstance(v, bool):
                if k in schema:
                    _, lo, hi, _ = schema[k]; v = min(max(v, lo), hi)
                out["rules"][rid][k] = type(out["rules"][rid][k])(v)
    return out


def load_settings(path=SETTINGS):
    try:
        with open(path, encoding="utf-8") as f:
            return validate_settings(json.load(f))
    except (OSError, ValueError):
        return defaults()


def save_settings(s, path=SETTINGS):
    s = validate_settings(s)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    return s
