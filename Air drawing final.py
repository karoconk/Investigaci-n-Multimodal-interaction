import cv2
import mediapipe as mp
import numpy as np
import math
from pathlib import Path
import pygame
pygame.mixer.init()

#MediaPipe 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

#Paleta de colores (en BGR)
COLORS = [
    (54,  67, 255), #Rojo
    (45, 255, 255), #Amarillo
    (50, 205,  50), #Verde
    (0, 165, 255),  #Naranja
    (150, 0, 27),   #Azul
]
color_index = 0
color_cooldown = 0

#Sonidos segun color
COLOR_SOUNDS = {
    0: pygame.mixer.Sound("sonidos/rojo.mp3"),
    1: pygame.mixer.Sound("sonidos/amarillo.mp3"),
    2: pygame.mixer.Sound("sonidos/verde.mp3"),
    3: pygame.mixer.Sound("sonidos/naranja.mp3"),
    4: pygame.mixer.Sound("sonidos/azul.mp3"),
}
#sonidos borrar
ACTION_SOUNDS = {
    "erase": pygame.mixer.Sound("sonidos/delete.wav"),
}

#Colores Cards instrucciones
C_BG    = (30,  18,  15)
C_GLASS = (60,  40,  35)
C_WHITE = (255, 255, 255)
C_RED   = (90, 100, 255)

GESTURE_DEFS = [
    {"label": "Dibujar", "sub": "1 dedo",      "color": (0, 210, 255), "key": "draw"},
    {"label": "Cambia color",   "sub": "2 dedos",     "color": (255, 190,  80), "key": "color"},
    {"label": "Mover",   "sub": "Pinza",       "color": (90, 255, 160), "key": "drag"},
    {"label": "Borrar",  "sub": "Mano abierta","color": (90, 100, 255), "key": "erase"},
]

HUD_H   = 130
CARD_W  = 140
CARD_GAP = 10
CARD_R  = 14
ICON_SIZE = 70   

#iconos cards
ICON_FILES = {
    "draw":  "img/draw.png",
    "color": "img/color.png",
    "drag":  "img/drag.png",
    "erase": "img/erase.png",
}

def load_icons(size=ICON_SIZE):
    icons = {}

    for key, filename in ICON_FILES.items():
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"No se pudo cargar {filename}")
            icons[key] = None
            continue

        img = cv2.resize(img, (size, size))

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        icons[key] = img

    return icons

ICONS = load_icons()

def paste_icon(bg, icon, cx, cy):
    ih, iw = icon.shape[:2]
    x, y = cx - iw // 2, cy - ih // 2
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + iw, bg.shape[1]), min(y + ih, bg.shape[0])
    ix1, iy1 = x1 - x, y1 - y
    ix2, iy2 = ix1 + (x2 - x1), iy1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return
    alpha = icon[iy1:iy2, ix1:ix2, 3:4].astype(np.float32) / 255.0
    fg    = icon[iy1:iy2, ix1:ix2, :3].astype(np.float32)
    roi   = bg[y1:y2, x1:x2].astype(np.float32)
    bg[y1:y2, x1:x2] = (fg * alpha + roi * (1 - alpha)).astype(np.uint8)


#Rectangulo para instrucciones
def rounded_rect_pts(x, y, w, h, r, steps=8):

    pts = []
    corners = [
        (x+r,   y+r,   np.pi,   1.5*np.pi),   # arriba-izquierda
        (x+w-r, y+r,   1.5*np.pi, 2*np.pi),   # arriba-derecha
        (x+w-r, y+h-r, 0,       0.5*np.pi),   # abajo-derecha
        (x+r,   y+h-r, 0.5*np.pi, np.pi),     # abajo-izquierda
    ]
    for cx2, cy2, a_start, a_end in corners:
        for t in np.linspace(a_start, a_end, steps):
            pts.append([int(cx2 + r*np.cos(t)), int(cy2 + r*np.sin(t))])
    return np.array(pts, dtype=np.int32)

def rounded_rect(img, x, y, w, h, r, color, thickness=-1, alpha=1.0):
    pts = rounded_rect_pts(x, y, w, h, r)
    if alpha < 1.0:
        overlay = img.copy()
        if thickness == -1:
            cv2.fillPoly(overlay, [pts], color)
        else:
            cv2.polylines(overlay, [pts], True, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    else:
        if thickness == -1:
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.polylines(img, [pts], True, color, thickness)

def build_hud(frame_w, frame_h, active_gesture=None):
    hud  = np.zeros((HUD_H, frame_w, 4), dtype=np.uint8)
    bg = np.full((HUD_H, frame_w, 3), C_BG, dtype=np.uint8)

    n     = len(GESTURE_DEFS)
    total = n * CARD_W + (n-1) * CARD_GAP
    sx    = (frame_w - total) // 2

    for i, gdef in enumerate(GESTURE_DEFS):
        x = sx + i * (CARD_W + CARD_GAP)
        y = 8
        w = CARD_W
        h = HUD_H - 16
        is_active = (active_gesture == gdef["key"])
        col   = gdef["color"] if is_active else C_GLASS
        alpha = 0.85 if is_active else 0.5

        rounded_rect(bg, x, y, w, h, CARD_R, col, -1, alpha)
        rounded_rect(bg, x, y, w, h, CARD_R,
                     gdef["color"] if is_active else (80,60,55), 2)

        icon_cx = x + w // 2
        icon_cy = y + 40

        icon_img = ICONS.get(gdef["key"])
        if icon_img is not None:
            paste_icon(bg, icon_img, icon_cx, icon_cy)

        (lw, _), _ = cv2.getTextSize(gdef["label"], cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
        cv2.putText(bg, gdef["label"],
                    (x + (w-lw)//2, y+h-24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, C_WHITE, 1, cv2.LINE_AA)

        (sw, _), _ = cv2.getTextSize(gdef["sub"], cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(bg, gdef["sub"],
                    (x + (w-sw)//2, y+h-9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    gdef["color"] if is_active else (180,160,150), 1, cv2.LINE_AA)

    hud[:,:,:3] = bg
    hud[:,:, 3] = 200
    return hud

def draw_color_swatch(frame, color, color_idx):
    cx, cy, r = 48, 48, 30
    cv2.circle(frame, (cx, cy), r+4, (50,40,38), -1)
    cv2.circle(frame, (cx, cy), r,   color,      -1)
    cv2.circle(frame, (cx, cy), r,   C_WHITE,     2)
    for i, c in enumerate(COLORS):
        dot_x = 16 + i * 12
        dot_r = 5 if i == color_idx else 3
        cv2.circle(frame, (dot_x, 90), dot_r, c, -1)
        if i == color_idx:
            cv2.circle(frame, (dot_x, 90), dot_r, C_WHITE, 1)
    cv2.putText(frame, "COLOR", (16, 108),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, C_WHITE, 1, cv2.LINE_AA)

def overlay_hud(frame, hud):
    h, w = frame.shape[:2]
    hud_h = hud.shape[0]
    y0  = h - hud_h
    roi = frame[y0:h, 0:w]
    alpha   = hud[:,:,3:4].astype(np.float32) / 255.0
    hud_bgr = hud[:,:,:3].astype(np.float32)
    roi_f   = roi.astype(np.float32)
    frame[y0:h, 0:w] = (hud_bgr * alpha + roi_f * (1-alpha)).astype(np.uint8)

#Lógica de gestos
strokes, current_stroke = [], []
missing_draw_frames = 0
dragging_stroke_index = None
last_drag_point = None
last_gesture = None

def draw_stroke(canvas, points, color, thickness=5):
    for i in range(1, len(points)):
        cv2.line(canvas, points[i-1], points[i], color, thickness)

def redraw_canvas(shape):
    canvas = np.zeros(shape, dtype=np.uint8)
    for s in strokes:
        draw_stroke(canvas, s["points"], s["color"])
    if current_stroke:
        draw_stroke(canvas, current_stroke, COLORS[color_index])
    return canvas

def commit_current_stroke():
    global current_stroke, missing_draw_frames
    if len(current_stroke) > 1:
        strokes.append({"points": current_stroke.copy(), "color": COLORS[color_index]})
    current_stroke = []
    missing_draw_frames = 0

def append_point_smoothly(pts, point, max_step=12):
    if not pts:
        pts.append(point); return
    lx, ly = pts[-1]; nx, ny = point
    d = math.hypot(nx-lx, ny-ly)
    if d <= max_step:
        pts.append(point); return
    steps = max(2, int(d / max_step))
    for step in range(1, steps+1):
        pts.append((int(lx+(nx-lx)*step/steps), int(ly+(ny-ly)*step/steps)))

def hand_scale(lms, shape):
    h, w, _ = shape
    wr = lms.landmark[0]; mid = lms.landmark[9]
    return max(1.0, math.hypot((mid.x-wr.x)*w, (mid.y-wr.y)*h))

def is_pinch(lms, shape):
    h, w, _ = shape
    th = lms.landmark[4]; ix = lms.landmark[8]
    d = math.hypot((th.x-ix.x)*w, (th.y-ix.y)*h)
    return d / hand_scale(lms, shape) < 0.45

def count_fingers(lms, hand_label=None):
    f = 0
    tt = lms.landmark[4]; ti = lms.landmark[3]
    if hand_label == "Right":
        if tt.x < ti.x: f += 1
    elif hand_label == "Left":
        if tt.x > ti.x: f += 1
    else:
        if abs(tt.x - ti.x) > 0.03: f += 1
    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        if lms.landmark[tip].y < lms.landmark[pip].y: f += 1
    return f

def detect_gesture(lms, shape, hand_label=None):
    if is_pinch(lms, shape): return "drag"
    fc = count_fingers(lms, hand_label)
    if fc == 1: return "draw"
    if fc == 2: return "color"
    if fc >= 4: return "erase"
    return "idle"

def hand_point(lms, shape):
    h, w, _ = shape
    lm = lms.landmark[8]
    return int(lm.x*w), int(lm.y*h)

def stroke_dist(pts, point):
    if not pts: return float("inf")
    tx, ty = point
    return min(math.hypot(px-tx, py-ty) for px, py in pts)

def find_nearest(point, max_d=70):
    ni, nd = None, max_d
    for i, s in enumerate(strokes):
        d = stroke_dist(s["points"], point)
        if d < nd: nd = d; ni = i
    return ni

def erase_nearest(point):
    i = find_nearest(point)
    if i is not None: del strokes[i]

def move_stroke(stroke, dx, dy):
    stroke["points"] = [(px+dx, py+dy) for px, py in stroke["points"]]

def start_drag(point):
    global dragging_stroke_index, last_drag_point
    dragging_stroke_index = find_nearest(point)
    last_drag_point = point if dragging_stroke_index is not None else None

def update_drag(point):
    global last_drag_point
    if dragging_stroke_index is None or last_drag_point is None: return
    move_stroke(strokes[dragging_stroke_index],
                point[0]-last_drag_point[0], point[1]-last_drag_point[1])
    last_drag_point = point

def stop_drag():
    global dragging_stroke_index, last_drag_point
    dragging_stroke_index = None; last_drag_point = None

def handle_gesture(gesture, point):
    global current_stroke, missing_draw_frames, color_index, last_gesture
    if gesture != "drag": stop_drag()
    if gesture == "draw":
        append_point_smoothly(current_stroke, point)
        missing_draw_frames = 0; last_gesture = gesture; return
    if current_stroke:
        missing_draw_frames += 1
        if missing_draw_frames > 6: commit_current_stroke()

    if gesture == "drag":
        if last_gesture != "drag":
            commit_current_stroke(); start_drag(point)
        else: update_drag(point)

    elif gesture == "color":
        global color_cooldown
        color_cooldown += 1
        if color_cooldown == 8:   # requiere ~8 frames sostenido (~0.25s) para cambiar
            commit_current_stroke()
            color_index = (color_index + 1) % len(COLORS)
            sound = COLOR_SOUNDS.get(color_index)
            if sound is not None:
                sound.play()
        if gesture != last_gesture:
            color_cooldown = 0

    elif gesture == "erase" and last_gesture != "erase":
        commit_current_stroke(); erase_nearest(point)
        s = ACTION_SOUNDS.get("erase")                    # ← al borrar
        if s is not None: s.play()

    if gesture != "color":
        color_cooldown = 0
    last_gesture = gesture
#Loop
cv2.namedWindow("Air Drawing", cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success: continue
    img = cv2.flip(img, 1)

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    active_gesture = None

    if not results.multi_hand_landmarks:
        stop_drag()
        if current_stroke:
            missing_draw_frames += 1
            if missing_draw_frames > 6: commit_current_stroke()

    if results.multi_hand_landmarks:
        for idx, lms in enumerate(results.multi_hand_landmarks):
            label = None
            if results.multi_handedness and idx < len(results.multi_handedness):
                label = results.multi_handedness[idx].classification[0].label
            point   = hand_point(lms, img.shape)
            gesture = detect_gesture(lms, img.shape, label)
            active_gesture = gesture
            handle_gesture(gesture, point)
            mp_draw.draw_landmarks(img, lms, mp_hands.HAND_CONNECTIONS)

    #Trazos
    canvas = redraw_canvas(img.shape)
    gray   = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    inv    = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    img    = cv2.bitwise_or(cv2.bitwise_and(img, inv), canvas)

    h, w = img.shape[:2]
    overlay_hud(img, build_hud(w, h, active_gesture))
    draw_color_swatch(img, COLORS[color_index], color_index)

    cv2.imshow("Air Drawing", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()