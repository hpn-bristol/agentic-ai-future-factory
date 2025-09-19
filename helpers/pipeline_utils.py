import re
import config

_DASHES = dict.fromkeys(map(ord, "‑–—"), "-")
_NBSP = {0xA0: ord(' ')}


def _canon(name: str) -> str:
    txt = name.translate(_DASHES).translate(_NBSP).strip().lower()
    return config._CANON.get(txt, txt)


_NUMLINE = re.compile(r"[\d\.]+\s*(.+)")


def _parse(text: str):
    mods = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.lower().startswith("candidate"):
            continue
        m = _NUMLINE.match(ln)
        raw = m.group(1).rstrip(".") if m else ln
        mods.append(_canon(raw))
    return mods


def pipe_key(pipeline_txt: str) -> str:
    mods = _parse(pipeline_txt)
    return " > ".join(mods)


_SPLIT = re.compile(r"Candidate-\d+[^\n\r]*?:", re.I)


def split_cands(raw: str) -> dict[str, str]:
    parts = _SPLIT.split(raw)
    tags = _SPLIT.findall(raw)
    if not tags:
        return {"cand_1": raw.strip()}
    return {tag.rstrip(':').strip(): body.strip()
            for tag, body in zip(tags, parts[1:])}
