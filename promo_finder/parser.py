"""Parse Kuku-style microdrama screenplays into structured, timestamped beats.

Pure code, no AI: this is the 'source of truth' that every LLM answer is checked against.
Each beat gets a stable ID like E03-B017 so the model can only point at real moments.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = r"(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?"
BLOCK_HDR = re.compile(r"^\*\*(.+?):\*\*\s*$")
START_END = re.compile(rf"^Start:\s*\({TS}\)\s*\|\s*End:\s*\({TS}\)")
SCENE_HDR = re.compile(r"^SCENE\s+(\d+)")
EP_NUM = re.compile(r"^EPISODE NUMBER:\s*(.*)$", re.I)
LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.*)$", re.I)
CONTINUITY = re.compile(r"^CONTINUITY NOTES:\s*(.*)$", re.I)
FILE_EP = re.compile(r"episode-(\d+)", re.I)


def to_sec(m: str, s: str, frac: str | None) -> float:
    return int(m) * 60 + int(s) + (float("0." + frac) if frac else 0.0)


def fmt(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Beat:
    id: str
    episode: int
    scene: int
    kind: str          # CAMERA | ACTION | SOUND FX | DIALOGUE
    speaker: str       # for DIALOGUE, else ""
    delivery: str      # e.g. "Cold, Commanding"
    start: float
    end: float
    text: str

    def line(self, include_camera: bool = True) -> str:
        """Compact one-line form sent to the LLM."""
        t = f"[{self.id} {fmt(self.start)}-{fmt(self.end)}]"
        if self.kind == "DIALOGUE":
            d = f" ({self.delivery})" if self.delivery else ""
            return f'{t} {self.speaker}{d}: {self.text}'
        return f"{t} <{self.kind}> {self.text}"


@dataclass
class Episode:
    number: int
    title: str
    logline: str
    continuity: str
    beats: list[Beat] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((b.end for b in self.beats), default=0.0)

    def transcript(self, include_camera: bool = True) -> str:
        return "\n".join(
            b.line() for b in self.beats if include_camera or b.kind != "CAMERA"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


def parse_episode(text: str, fallback_number: int) -> Episode:
    lines = text.splitlines()
    number, logline, continuity, title = fallback_number, "", "", ""
    for ln in lines[:10]:
        if m := EP_NUM.match(ln):
            title = m.group(1).strip()
            if n := re.search(r"\d+", title):
                number = int(n.group())
        elif m := LOGLINE.match(ln):
            logline = m.group(1).strip()
        elif m := CONTINUITY.match(ln):
            continuity = m.group(1).strip()

    ep = Episode(number=number, title=title or f"Episode {number}",
                 logline=logline, continuity=continuity)
    scene, i, k = 0, 0, 0
    while i < len(lines):
        ln = lines[i].strip()
        if m := SCENE_HDR.match(ln):
            scene = int(m.group(1))
        elif m := BLOCK_HDR.match(ln):
            header = m.group(1).strip()
            # next non-empty line should be Start/End
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            se = START_END.match(lines[j].strip()) if j < len(lines) else None
            if se:
                g = se.groups()
                start, end = to_sec(*g[0:3]), to_sec(*g[3:6])
                body, j = [], j + 1
                while j < len(lines) and lines[j].strip() and not BLOCK_HDR.match(lines[j].strip()) \
                        and not lines[j].startswith("---"):
                    body.append(lines[j].strip())
                    j += 1
                txt = " ".join(body).strip()
                up = header.upper()
                if up in ("CAMERA", "ACTION", "SOUND FX", "SFX", "MUSIC", "VO"):
                    kind, speaker, delivery = ("SOUND FX" if up == "SFX" else up), "", ""
                else:
                    kind = "DIALOGUE"
                    # "LADY ROSE (Cold, Commanding)" / "MATRIARCH (O.S.) (Firm)"
                    parens = re.findall(r"\(([^)]*)\)", header)
                    speaker = re.sub(r"\s*\(.*$", "", header).strip()
                    delivery = ", ".join(p for p in parens if p.upper() not in ("O.S.", "V.O.", "CONT'D"))
                    if any(p.upper() in ("O.S.", "V.O.") for p in parens):
                        speaker += " (O.S.)"
                    txt = txt.strip('"“”')
                k += 1
                ep.beats.append(Beat(
                    id=f"E{number:02d}-B{k:03d}", episode=number, scene=scene,
                    kind=kind, speaker=speaker, delivery=delivery,
                    start=start, end=end, text=txt))
                i = j
                continue
        i += 1
    ep.beats.sort(key=lambda b: (b.start, b.end))
    return ep


def _ep_num_from_name(name: str, default: int) -> int:
    m = FILE_EP.search(Path(name).name)
    return int(m.group(1)) if m else default


def load_show(path: str | Path | None = None, files: dict[str, bytes] | None = None) -> list[Episode]:
    """Load a show from a folder, a .zip, or an in-memory {name: bytes} dict (Streamlit upload)."""
    raw: dict[str, str] = {}
    if files is not None:
        for name, data in files.items():
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for n in z.namelist():
                        if n.lower().endswith(".txt") and not n.startswith("__MACOSX"):
                            raw[n] = z.read(n).decode("utf-8", "ignore")
            elif name.lower().endswith(".txt"):
                raw[name] = data.decode("utf-8", "ignore")
    else:
        p = Path(path)
        if p.suffix.lower() == ".zip":
            return load_show(files={p.name: p.read_bytes()})
        for f in sorted(p.rglob("*.txt")):
            raw[str(f)] = f.read_text("utf-8", "ignore")

    eps = [parse_episode(t, _ep_num_from_name(n, i + 1))
           for i, (n, t) in enumerate(sorted(raw.items()))]
    eps = [e for e in eps if e.beats]
    eps.sort(key=lambda e: e.number)
    return eps


def beat_index(episodes: list[Episode]) -> dict[str, Beat]:
    return {b.id: b for e in episodes for b in e.beats}
