"""Rule registry. Every rule reads the compiled SceneState and returns Alarms."""
from dataclasses import dataclass, field

REGISTRY = {}


@dataclass
class Alarm:
    rule_id: str
    type: str                 # Russian violation name shown to the jury
    description: str
    timecode_sec: float
    start: float = None
    end: float = None
    evidence: dict = field(default_factory=dict)   # debug facts that made the rule fire

    def as_dict(self):
        return {"rule_id": self.rule_id, "type": self.type, "description": self.description,
                "timecode_sec": round(self.timecode_sec, 1),
                "interval_sec": [round(self.start if self.start is not None else self.timecode_sec, 1),
                                 round(self.end if self.end is not None else self.timecode_sec, 1)],
                "evidence": self.evidence}


class Rule:
    rule_id = ""
    type = ""
    description = ""
    class_id = ""             # violation_classes.csv row with the legal citation
    params = {}               # defaults; settings.json may override per rule
    param_schema = {}         # name -> (label, min, max, step)

    def evaluate(self, timeline, params, scene):
        raise NotImplementedError

    def alarm(self, t, start=None, end=None, description=None, **evidence):
        return Alarm(self.rule_id, self.type, description or self.description, t, start, end, evidence)


def register(cls):
    REGISTRY[cls.rule_id] = cls()
    return cls


def evaluate_rules(timeline, scene, rule_settings):
    """rule_settings: {rule_id: {"enabled": bool, **params}} (missing -> defaults)."""
    from app.rules import violations  # noqa: F401  (registers every rule)
    out = []
    for rid, rule in REGISTRY.items():
        cfg = {**rule.params, **(rule_settings.get(rid) or {})}
        if not cfg.pop("enabled", True):
            continue
        out.extend(rule.evaluate(timeline, cfg, scene))
    return sorted(out, key=lambda a: a.timecode_sec)
