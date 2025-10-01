
# main.py — processamento/visualização (VERSÃO MODIFICADA)
# Alterações principais:
# - Setas do teclado para mover o jogador (↑ ↓ ← →)
# - Agentes desenhados como quadrados (cada track tem cor fixa)
# - Trilhas por track (últimos TRAIL_LENGTH pontos)
# Mantive os demais comportamentos (C/L/E toggles, respawn, pause, etc.)
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "glut")

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

from loaders import parse_paths_file, load_groups_data
import gc

# ===== Config =====
WINDOW_W, WINDOW_H = 1280, 720
AGENT_POINT_SIZE = 6.0     # tamanho usado para o quadrado (px)
COLOR_D = (0.2, 0.8, 1.0)   # fallback (Paths_D)
COLOR_N = (1.0, 0.6, 0.2)   # fallback (Paths_N)
HIGHLIGHT_COLOR = (1.0, 0.2, 0.2)  # pontos próximos (vermelho)

# Proximidade / repulsão
PROXIMITY_RADIUS = 20.0     # raio p/ considerar "muito perto" (px)
LINK_ALPHA = 0.25           # alpha das linhas entre pares próximos
AVOID_RADIUS = 18.0         # raio de repulsão (px)
AVOID_STRENGTH = 0.6        # força da repulsão (multiplicador)
AVOID_MAX_DISP = 8.0        # deslocamento máximo por frame (px)

# Trilhas
TRAIL_LENGTH = 30           # quantos pontos atrás desenhar como trilha
TRAIL_LINE_WIDTH = 2.0

# Loop/tempo
RENDER_FPS_TARGET = 60.0
LOG_FPS = False

# ===== Estado =====
@dataclass
class Player:
    x: float
    y: float
    alive: bool = True

play = True                 # animação dos frames
frame_rate = 30.0           # fps lógico dos dados
frame_idx = 0
max_frame = 0

data_dir = Path("data")
# agora cada frame mapeará para uma lista de (x, y, track_id)
frame_points_D: Dict[int, List[Tuple[float, float, int]]] = {}
frame_points_N: Dict[int, List[Tuple[float, float, int]]] = {}
groups: Dict[int, List[float]] = {}

# toggles de processamento
COLOR_NEAR = True
LINK_NEAR = False
AVOIDANCE_ENABLED = False

# teclado (agora usando setas: up/down/left/right)
_keys_down: Dict[str, bool] = {"left": False, "right": False, "up": False, "down": False}

# boneco
PLAYER_COLOR = (1.0, 1.0, 0.2)
PLAYER_SIZE = 10.0
PLAYER_SPEED = 220.0
PLAYER_HIT_RADIUS = 10.0
player = Player(x=WINDOW_W * 0.5, y=WINDOW_H * 0.5, alive=True)

# pontos do frame atual (para colisão/respawn do boneco)
_last_pts_all_np = np.empty((0, 2), dtype=np.float32)

# cores por track (será preenchido ao carregar dados)
_colors_per_track = np.empty((0,3), dtype=np.float32)
# lista por track: cada elemento é lista de (frame, x, y)
_tracks_points_by_id: List[List[Tuple[int, float, float]]] = []

# ===== Utils =====
def clamp(v, vmin, vmax): return max(vmin, min(v, vmax))

def move_player(dt: float):
    if not player.alive: return
    vx = vy = 0.0
    if _keys_down.get("left"): vx -= PLAYER_SPEED
    if _keys_down.get("right"): vx += PLAYER_SPEED
    if _keys_down.get("up"): vy -= PLAYER_SPEED
    if _keys_down.get("down"): vy += PLAYER_SPEED
    player.x = clamp(player.x + vx*dt, PLAYER_SIZE, WINDOW_W - PLAYER_SIZE)
    player.y = clamp(player.y + vy*dt, PLAYER_SIZE, WINDOW_H - PLAYER_SIZE)

def update_last_pts_np(pts_all):
    global _last_pts_all_np
    _last_pts_all_np = np.asarray(pts_all, dtype=np.float32) if pts_all else np.empty((0,2), np.float32)

def collides_with_points(px: float, py: float, radius: float) -> bool:
    if _last_pts_all_np.size == 0: return False
    dx = _last_pts_all_np[:,0] - px
    dy = _last_pts_all_np[:,1] - py
    return np.any(dx*dx + dy*dy <= radius*radius)

def check_collision_player():
    if player.alive and collides_with_points(player.x, player.y, PLAYER_HIT_RADIUS):
        player.alive = False

def respawn_player():
    """Respawn no centro (busca posição livre ao redor)."""
    cx, cy = WINDOW_W*0.5, WINDOW_H*0.5
    candidates = [(0,0)]
    for r in (15, 30, 45, 60, 90, 120, 160, 200):
        candidates += [( r, 0), (-r, 0), (0, r), (0,-r),
                       ( r, r), (-r, r), ( r,-r), (-r,-r)]
    for dx, dy in candidates:
        px = clamp(cx + dx, PLAYER_SIZE, WINDOW_W - PLAYER_SIZE)
        py = clamp(cy + dy, PLAYER_SIZE, WINDOW_H - PLAYER_SIZE)
        if not collides_with_points(px, py, PLAYER_HIT_RADIUS):
            player.x, player.y, player.alive = px, py, True
            return
    player.x, player.y, player.alive = cx, cy, True

# ===== GL draws =====
def draw_agents_as_quads(P: np.ndarray, colors: np.ndarray, size_px: float):
    """Desenha cada agente como um quadrado centralizado em P[i]."""
    if P.size == 0: return
    half = size_px * 0.5
    glBegin(GL_QUADS)
    for i in range(P.shape[0]):
        r, g, b = float(colors[i,0]), float(colors[i,1]), float(colors[i,2])
        glColor3f(r, g, b)
        x, y = float(P[i,0]), float(P[i,1])
        glVertex2f(x - half, y - half)
        glVertex2f(x + half, y - half)
        glVertex2f(x + half, y + half)
        glVertex2f(x - half, y + half)
    glEnd()

def draw_lines_pairs(P: np.ndarray, pairs_ij: np.ndarray, rgb=(1,1,1), alpha=0.2):
    if pairs_ij.size == 0: return
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(rgb[0], rgb[1], rgb[2], alpha)
    glBegin(GL_LINES)
    for i, j in pairs_ij:
        glVertex2f(P[i,0], P[i,1])
        glVertex2f(P[j,0], P[j,1])
    glEnd()
    glDisable(GL_BLEND)

def draw_player():
    if not player.alive: return
    glColor3f(*PLAYER_COLOR)
    h = PLAYER_SIZE
    glBegin(GL_QUADS)
    glVertex2f(player.x - h, player.y - h)
    glVertex2f(player.x + h, player.y - h)
    glVertex2f(player.x + h, player.y + h)
    glVertex2f(player.x - h, player.y + h)
    glEnd()

def draw_text(x, y, msg):
    glColor3f(1,1,1); glRasterPos2f(x, y)
    for ch in msg: glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(ch))

def draw_trail_for_track(track_id: int, upto_frame: int):
    """Desenha a trilha do track_id até o frame 'upto_frame'.
       Usa _tracks_points_by_id global. """
    pts = _tracks_points_by_id[track_id] if 0 <= track_id < len(_tracks_points_by_id) else []
    if not pts:
        return
    # coletar posições com frame <= upto_frame
    selected = [ (x,y) for (f,x,y) in pts if f <= upto_frame ]
    if not selected:
        return
    # pegar últimos TRAIL_LENGTH pontos
    trail = np.asarray(selected[-TRAIL_LENGTH:], dtype=np.float32)
    if trail.shape[0] < 2: return
    base_color = _colors_per_track[track_id % _colors_per_track.shape[0]]
    # desenhar com fade (mais recente mais opaco)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(TRAIL_LINE_WIDTH)
    n = trail.shape[0]
    for i in range(n-1):
        a = (i+1)/n  # alpha crescente ao longo da trilha
        r,g,b = base_color
        glColor4f(float(r), float(g), float(b), 0.15 + 0.85*a)
        glBegin(GL_LINES)
        glVertex2f(trail[i,0], trail[i,1])
        glVertex2f(trail[i+1,0], trail[i+1,1])
        glEnd()
    glDisable(GL_BLEND)

# ===== Processamento por frame =====
def build_frame_index(tracks_raw, track_offset=0) -> Dict[int, List[Tuple[float, float, int]]]:
    """Retorna dict: frame -> list de (x, y, track_id)."""
    idx: Dict[int, List[Tuple[float, float, int]]] = {}
    for ti, tr in enumerate(tracks_raw):
        track_id = track_offset + ti
        for x, y, f in tr["points"]:
            idx.setdefault(int(f), []).append((float(x), float(y), track_id))
    return idx

def proximity_analysis(P: np.ndarray, radius: float):
    """Retorna (mask_near_any, pairs_ij) usando distância euclidiana."""
    n = P.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool), np.empty((0,2), dtype=np.int32)
    dx = P[:,0][:,None] - P[:,0][None,:]
    dy = P[:,1][:,None] - P[:,1][None,:]
    d2 = dx*dx + dy*dy
    near = (d2 > 0.0) & (d2 <= radius*radius)
    mask_any = near.any(axis=1)
    iu, ju = np.where(np.triu(near, 1))
    pairs = np.stack([iu, ju], axis=1) if iu.size else np.empty((0,2), dtype=np.int32)
    return mask_any, pairs

def avoidance_displacement(P: np.ndarray, radius: float, strength: float, max_disp: float):
    """Repulsão: deslocamento acumulado longe dos vizinhos dentro do raio."""
    n = P.shape[0]
    if n == 0: return np.zeros((0,2), dtype=np.float32)
    dx = P[:,0][:,None] - P[:,0][None,:]
    dy = P[:,1][:,None] - P[:,1][None,:]
    d2 = dx*dx + dy*dy
    mask = (d2 > 0.0) & (d2 <= radius*radius)
    eps = 1e-5
    inv = np.where(mask, 1.0/(d2 + eps), 0.0)
    vx = (dx * inv).sum(axis=1)
    vy = (dy * inv).sum(axis=1)
    disp = np.stack([vx, vy], axis=1) * strength
    mag = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-8
    scale = np.minimum(1.0, max_disp / mag)
    disp = disp * scale
    return disp.astype(np.float32)

# ===== Render do frame =====
def render_frame():
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluOrtho2D(0, WINDOW_W, WINDOW_H, 0)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT)

    f = frame_idx + 1
    PD = frame_points_D.get(f, [])
    PN = frame_points_N.get(f, [])
    # extrair coords e track_ids se existirem
    AD = np.asarray([(x,y) for (x,y,tid) in PD], dtype=np.float32) if PD else np.empty((0,2), np.float32)
    AD_tids = np.asarray([tid for (x,y,tid) in PD], dtype=np.int32) if PD else np.empty((0,), np.int32)
    AN = np.asarray([(x,y) for (x,y,tid) in PN], dtype=np.float32) if PN else np.empty((0,2), np.float32)
    AN_tids = np.asarray([tid for (x,y,tid) in PN], dtype=np.int32) if PN else np.empty((0,), np.int32)
    P = np.vstack([AD, AN]) if (AD.size or AN.size) else np.empty((0,2), np.float32)
    track_ids = np.concatenate([AD_tids, AN_tids]) if (AD_tids.size or AN_tids.size) else np.empty((0,), np.int32)
    nD, nN = AD.shape[0], AN.shape[0]
    n = P.shape[0]

    if AVOIDANCE_ENABLED and n > 0:
        disp = avoidance_displacement(P, AVOID_RADIUS, AVOID_STRENGTH, AVOID_MAX_DISP)
        P = P + disp
        P[:,0] = np.clip(P[:,0], 0, WINDOW_W)
        P[:,1] = np.clip(P[:,1], 0, WINDOW_H)

    if n > 0:
        near_mask, pairs = proximity_analysis(P, PROXIMITY_RADIUS)
    else:
        near_mask = np.zeros((0,), bool); pairs = np.empty((0,2), np.int32)

    # cores por agente (consistentes por track)
    if n > 0 and _colors_per_track.size != 0:
        colors = _colors_per_track[track_ids % _colors_per_track.shape[0]]
        # highlight para agentes próximos
        if COLOR_NEAR:
            colors = colors.copy()
            colors[near_mask,:] = HIGHLIGHT_COLOR
    else:
        colors = np.empty((0,3), np.float32)

    # atualizar pontos para colisão/respawn
    update_last_pts_np(P.tolist())
    check_collision_player()

    # desenhar trilhas para tracks presentes no frame (limitado)
    unique_tids = np.unique(track_ids) if track_ids.size else np.empty((0,), np.int32)
    for tid in unique_tids:
        draw_trail_for_track(int(tid), frame_idx + 1)

    if LINK_NEAR and pairs.size:
        draw_lines_pairs(P, pairs, rgb=(1,1,1), alpha=LINK_ALPHA)
    draw_agents_as_quads(P, colors, AGENT_POINT_SIZE)

    if not play: draw_text(10, 20, "PAUSED (P)")
    draw_text(10, WINDOW_H-15, f"[E]{'on' if AVOIDANCE_ENABLED else 'off'} [C]{'on' if COLOR_NEAR else 'off'} [L]{'on' if LINK_NEAR else 'off'}")

    draw_player()
    glFlush()

# ===== Teclado =====
def _on_key_down(key: bytes, x: int, y: int):
    global play, COLOR_NEAR, LINK_NEAR, AVOIDANCE_ENABLED
    # mantive controle por letras para toggles/pausa/respawn/ESC
    try:
        k = key.decode("utf-8").lower()
    except Exception:
        k = ""
    if k == '\x1b': sys.exit(0)     # ESC
    if key == b' ': respawn_player()   # Space -> respawn
    if k == 'p':    global play; play = not play
    if k == 'c':    global COLOR_NEAR; COLOR_NEAR = not COLOR_NEAR
    if k == 'l':    global LINK_NEAR; LINK_NEAR = not LINK_NEAR
    if k == 'e':    global AVOIDANCE_ENABLED; AVOIDANCE_ENABLED = not AVOIDANCE_ENABLED  # toggle E

def _on_key_up(key: bytes, x: int, y: int):
    # tecla normal - não usada para movimento por setas
    pass

def _on_special_down(key, x, y):
    # GLUT special keys (setas)
    if key == GLUT_KEY_LEFT:  _keys_down["left"] = True
    if key == GLUT_KEY_RIGHT: _keys_down["right"] = True
    if key == GLUT_KEY_UP:    _keys_down["up"] = True
    if key == GLUT_KEY_DOWN:  _keys_down["down"] = True

def _on_special_up(key, x, y):
    if key == GLUT_KEY_LEFT:  _keys_down["left"] = False
    if key == GLUT_KEY_RIGHT: _keys_down["right"] = False
    if key == GLUT_KEY_UP:    _keys_down["up"] = False
    if key == GLUT_KEY_DOWN:  _keys_down["down"] = False

# ===== Helpers para cores =====
def hsv_to_rgb(h, s, v):
    """Converte HSV (0..1) -> RGB (0..1)."""
    i = int(h * 6.0)  # assume h in [0,1)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    return (v, p, q)

# ===== Main =====
def main():
    global data_dir, frame_points_D, frame_points_N, groups, max_frame
    global _colors_per_track, _tracks_points_by_id

    gc.disable()  # evita pausas longas do GC no loop

    if len(sys.argv) > 1:
        d = Path(sys.argv[1])
        if d.exists(): data_dir = d

    # carrega dados e indexa por frame (incluindo track_id)
    D_raw = parse_paths_file(data_dir / "Paths_D.txt")
    N_raw = parse_paths_file(data_dir / "Paths_N.txt")
    frame_points_D = build_frame_index(D_raw, track_offset=0)
    frame_points_N = build_frame_index(N_raw, track_offset=len(D_raw))
    groups = load_groups_data(data_dir / "CN-01-GROUPS_DATA.txt")
    max_frame = max(frame_points_D.keys() | frame_points_N.keys())

    # construir lista de pontos por track (frame,x,y)
    total_tracks = len(D_raw) + len(N_raw)
    _tracks_points_by_id = [ [] for _ in range(total_tracks) ]
    for ti, tr in enumerate(D_raw):
        for x, y, f in tr["points"]:
            _tracks_points_by_id[ti].append((int(f), float(x), float(y)))
    for ti, tr in enumerate(N_raw):
        gid = len(D_raw) + ti
        for x, y, f in tr["points"]:
            _tracks_points_by_id[gid].append((int(f), float(x), float(y)))
    # ordenar por frame (quando necessário)
    for lst in _tracks_points_by_id:
        lst.sort(key=lambda t: t[0])

    # gerar cor fixa por track (HSV distribuído)
    if total_tracks > 0:
        cols = []
        for i in range(total_tracks):
            h = (i / float(total_tracks)) * 0.85  # espalha no espectro
            s = 0.7
            v = 0.9
            cols.append(hsv_to_rgb(h, s, v))
        _colors_per_track = np.asarray(cols, dtype=np.float32)
    else:
        _colors_per_track = np.empty((0,3), np.float32)

    # GLUT
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(WINDOW_W, WINDOW_H)
    win = glutCreateWindow(b"T1 CG - Trilhas & Setas (modificado)")
    if not win:
        print("[ERRO] Nao foi possivel criar janela OpenGL."); sys.exit(1)
    glutSetWindow(win)
    glutDisplayFunc(render_frame)
    glDisable(GL_DEPTH_TEST)

    try:
        glutKeyboardFunc(_on_key_down)
        glutKeyboardUpFunc(_on_key_up)
        glutSpecialFunc(_on_special_down)
        # GLUT doesn't provide special up in all implementations; set via try/except
        try:
            glutSpecialUpFunc(_on_special_up)
        except Exception:
            pass
    except Exception:
        pass

    print(f"[INFO] Frames: 0..{max_frame}")
    print("[INFO] Setas move | SPACE respawn | P pause | C color-prox | L linhas | E evitar colisao | ESC sai")

    # loop manual com temporização estável
    last = time.perf_counter()
    acc_frames = 0.0
    render_period = 1.0 / RENDER_FPS_TARGET
    frames = 0; t0 = last

    while True:
        frame_start = time.perf_counter()
        glutMainLoopEvent()

        now = time.perf_counter()
        dt = now - last
        last = now

        move_player(dt)

        if play:
            acc_frames += dt * frame_rate
            steps = int(acc_frames)
            if steps:
                acc_frames -= steps
                global frame_idx
                frame_idx = (frame_idx + steps) % (max_frame + 1)

        render_frame()
        glutSwapBuffers()

        if LOG_FPS and (now - t0) > 1.0:
            print(f"[fps] ~{frames/(now - t0):.1f}")
            frames = 0; t0 = now
        else:
            frames += 1

        elapsed = time.perf_counter() - frame_start
        to_sleep = render_period - elapsed
        if to_sleep > 0:
            time.sleep(min(to_sleep, 0.002))

if __name__ == "__main__":
    main()
