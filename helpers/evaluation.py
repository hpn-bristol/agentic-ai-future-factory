import config
from .pipeline_utils import _parse


def evaluate(intent: str, cand_text: str):
    req = config.GOLD[intent]
    mods = _parse(cand_text)
    pos = {m: i for i, m in enumerate(mods)}

    present = set(mods)
    mandatory = set()
    optional = set()

    prev = -1
    for kind, grp in req:
        hit = present & grp

        if kind == "all":
            if hit != grp:
                return 0.0, "bad"
            mandatory |= grp

        elif kind == "any":
            if not hit:
                return 0.0, "bad"
            mandatory |= hit

        elif kind == "opt":
            optional |= hit

        if kind in {"all", "any"}:
            idx = [pos[m] for m in hit]
            if idx and min(idx) < prev:
                return 0.0, "bad"
            if idx:
                prev = max(idx)

    extra = present - mandatory - optional
    return (1.0, "perfect") if not extra else (0.5, "partial")
