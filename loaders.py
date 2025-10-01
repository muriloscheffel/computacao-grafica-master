from pathlib import Path
import re

def load_image_list(path: Path):
    """Lê imageList.txt com os nomes dos arquivos de imagem (um por linha)."""
    names = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                names.append(ln)
    return names

def parse_paths_file(path: Path):
    """
    Formato por linha: 'N (x,y,f)(x,y,f)...'
    Retorna: lista de tracks: {"count": N|len, "points": [(x,y,f), ...]}
    """
    tracks = []
    tuple_re = re.compile(r"\((\-?\d+\.?\d*),\s*(\-?\d+\.?\d*),\s*(\-?\d+)\)")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            try:
                n_declared = int(parts[0])
            except:
                n_declared = None
            tuples_part = parts[1] if len(parts) > 1 else line
            pts = []
            for m in tuple_re.finditer(tuples_part):
                x = float(m.group(1)); y = float(m.group(2)); fidx = int(float(m.group(3)))
                pts.append((x, y, fidx))
            if pts:
                tracks.append({"count": n_declared or len(pts), "points": pts})
    return tracks

def load_groups_data(path: Path):
    """
    CN-01-GROUPS_DATA.txt: col0=frame (1-based), col1=total, demais colunas variam.
    Retorna dict: frame -> [valores...]
    """
    data = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            cols = ln.split()
            try:
                frame = int(float(cols[0]))
            except:
                continue
            values = [try_number(c) for c in cols[1:]]
            data[frame] = values
    return data

def try_number(s: str):
    try:
        if "." in s: return float(s)
        return int(s)
    except:
        return s
