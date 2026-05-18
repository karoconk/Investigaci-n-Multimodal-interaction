import cv2
import mediapipe as mp
import numpy as np
import math
import time
import winsound
import os

# ============================================
# CONFIGURACIÓN DE SONIDOS PERSONALIZADOS
# ============================================

# Rutas de tus archivos de sonido (cambia los nombres por los tuyos)
SONIDO_COLOR = "sonidos/change.wav"    # Cambia por tu archivo
SONIDO_BORRAR = "sonidos/delete.wav"         # Cambia por tu archivo
SONIDO_ARRANQUE = "sonidos/drag.wav"    # Cambia por tu archivo

# Verificar qué sonidos existen
def verificar_sonidos():
    sonidos_disponibles = {
        'color': os.path.exists(SONIDO_COLOR),
        'erase': os.path.exists(SONIDO_BORRAR),
        'drag': os.path.exists(SONIDO_ARRANQUE)
    }
    
    print("\n🔊 Verificando archivos de sonido:")
    for nombre, existe in sonidos_disponibles.items():
        if existe:
            print(f"  ✅ {nombre}: encontrado")
        else:
            print(f"  ⚠️ {nombre}: NO encontrado - usando beep")
    
    return sonidos_disponibles

sonidos_existentes = verificar_sonidos()

def reproducir_sonido_accion(tipo):
    """Reproduce sonido personalizado si existe, si no usa beep"""
    if tipo == 'color' and sonidos_existentes['color']:
        winsound.PlaySound(SONIDO_COLOR, winsound.SND_FILENAME | winsound.SND_ASYNC)
    elif tipo == 'erase' and sonidos_existentes['erase']:
        winsound.PlaySound(SONIDO_BORRAR, winsound.SND_FILENAME | winsound.SND_ASYNC)
    elif tipo == 'drag' and sonidos_existentes['drag']:
        winsound.PlaySound(SONIDO_ARRANQUE, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        # Fallback a beep si no hay archivo
        if tipo == 'color':
            winsound.Beep(1046, 100)
        elif tipo == 'erase':
            winsound.Beep(330, 200)
        elif tipo == 'drag':
            winsound.Beep(660, 150)

# ============================================
# INICIALIZAR MEDIAPIPE
# ============================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara. Revisa que exista una webcam disponible.")

strokes = []
current_stroke = []
current_color_index = 0
current_color = None
missing_draw_frames = 0
dragging_stroke_index = None
last_drag_point = None
last_gesture = None

colors = [(0,0,255), (255,0,0), (0,255,0), (0,255,255)]
color_index = 0


def draw_stroke(canvas, stroke_points, stroke_color, thickness=5):
    for i in range(1, len(stroke_points)):
        cv2.line(canvas, stroke_points[i - 1], stroke_points[i], stroke_color, thickness)


def redraw_canvas(frame_shape):
    canvas = np.zeros(frame_shape, dtype=np.uint8)

    for stroke in strokes:
        draw_stroke(canvas, stroke["points"], stroke["color"])

    if current_stroke:
        draw_stroke(canvas, current_stroke, current_color)

    return canvas


def commit_current_stroke():
    global current_stroke, missing_draw_frames

    if len(current_stroke) > 1:
        strokes.append({"points": current_stroke.copy(), "color": current_color})

    current_stroke = []
    missing_draw_frames = 0


def append_point_smoothly(stroke_points, point, max_step=12):
    if not stroke_points:
        stroke_points.append(point)
        return

    last_x, last_y = stroke_points[-1]
    new_x, new_y = point
    distance = math.hypot(new_x - last_x, new_y - last_y)

    if distance <= max_step:
        stroke_points.append(point)
        return

    steps = max(2, int(distance / max_step))
    for step in range(1, steps + 1):
        x = int(last_x + (new_x - last_x) * step / steps)
        y = int(last_y + (new_y - last_y) * step / steps)
        stroke_points.append((x, y))


def stroke_bounds(stroke_points, margin=20):
    xs = [point[0] for point in stroke_points]
    ys = [point[1] for point in stroke_points]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def point_inside_bounds(x, y, bounds):
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2


def move_stroke(stroke, dx, dy):
    moved_points = [(point_x + dx, point_y + dy) for point_x, point_y in stroke["points"]]
    stroke["points"] = moved_points


def hand_point(handLms, img_shape):
    height, width, _ = img_shape
    index_tip = handLms.landmark[8]
    return int(index_tip.x * width), int(index_tip.y * height)


def pinch_point(handLms, img_shape):
    height, width, _ = img_shape
    thumb_tip = handLms.landmark[4]
    index_tip = handLms.landmark[8]
    return (
        int(((thumb_tip.x + index_tip.x) / 2) * width),
        int(((thumb_tip.y + index_tip.y) / 2) * height),
    )


def hand_scale(handLms, img_shape):
    height, width, _ = img_shape
    wrist = handLms.landmark[0]
    middle_mcp = handLms.landmark[9]
    return max(1.0, math.hypot((middle_mcp.x - wrist.x) * width, (middle_mcp.y - wrist.y) * height))


def is_pinch(handLms, img_shape):
    height, width, _ = img_shape
    thumb_tip = handLms.landmark[4]
    index_tip = handLms.landmark[8]
    pinch_distance = math.hypot((thumb_tip.x - index_tip.x) * width, (thumb_tip.y - index_tip.y) * height)
    return pinch_distance / hand_scale(handLms, img_shape) < 0.35


def count_fingers(handLms, hand_label=None):
    fingers = 0

    thumb_tip = handLms.landmark[4]
    thumb_ip = handLms.landmark[3]
    if hand_label == "Right":
        if thumb_tip.x < thumb_ip.x:
            fingers += 1
    elif hand_label == "Left":
        if thumb_tip.x > thumb_ip.x:
            fingers += 1
    else:
        if abs(thumb_tip.x - thumb_ip.x) > 0.03:
            fingers += 1

    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip, pip in finger_pairs:
        if handLms.landmark[tip].y < handLms.landmark[pip].y:
            fingers += 1

    return fingers


def detect_gesture(handLms, img_shape, hand_label=None):
    if is_pinch(handLms, img_shape):
        return "drag"

    finger_count = count_fingers(handLms, hand_label)

    if finger_count == 1:
        return "draw"
    if finger_count == 2:
        return "color"
    if finger_count >= 4:
        return "erase"
    return "idle"


def stroke_distance_to_point(stroke_points, point):
    if not stroke_points:
        return float("inf")

    target_x, target_y = point
    return min(math.hypot(point_x - target_x, point_y - target_y) for point_x, point_y in stroke_points)


def find_nearest_stroke_index(point, max_distance=70):
    nearest_index = None
    nearest_distance = max_distance

    for index, stroke in enumerate(strokes):
        distance = stroke_distance_to_point(stroke["points"], point)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index

    return nearest_index


def erase_nearest_stroke(point):
    nearest_index = find_nearest_stroke_index(point)
    if nearest_index is not None:
        del strokes[nearest_index]


def start_drag(point):
    global dragging_stroke_index, last_drag_point

    dragging_stroke_index = find_nearest_stroke_index(point)
    last_drag_point = point if dragging_stroke_index is not None else None


def update_drag(point):
    global last_drag_point

    if dragging_stroke_index is None or last_drag_point is None:
        return

    dx = point[0] - last_drag_point[0]
    dy = point[1] - last_drag_point[1]
    move_stroke(strokes[dragging_stroke_index], dx, dy)
    last_drag_point = point


def stop_drag():
    global dragging_stroke_index, last_drag_point

    dragging_stroke_index = None
    last_drag_point = None


def handle_gesture(gesture, point):
    global current_stroke, missing_draw_frames, color_index, current_color, last_gesture

    if gesture != "drag":
        stop_drag()

    if gesture == "draw":
        append_point_smoothly(current_stroke, point)
        missing_draw_frames = 0
        last_gesture = gesture
        return

    if current_stroke:
        missing_draw_frames += 1
        if missing_draw_frames > 6:
            commit_current_stroke()

    if gesture == "drag":
        if last_gesture != "drag":
            commit_current_stroke()
            start_drag(point)
            reproducir_sonido_accion('drag')  # Sonido personalizado o beep
        else:
            update_drag(point)

    elif gesture == "color" and last_gesture != "color":
        commit_current_stroke()
        color_index = (color_index + 1) % len(colors)
        current_color = colors[color_index]
        reproducir_sonido_accion('color')  # Sonido personalizado o beep

    elif gesture == "erase" and last_gesture != "erase":
        commit_current_stroke()
        erase_nearest_stroke(point)
        reproducir_sonido_accion('erase')  # Sonido personalizado o beep

    last_gesture = gesture


current_color = colors[color_index]

running = True
cv2.namedWindow("Air Drawing PRO")

print("=" * 50)
print("🎨 Air Drawing PRO con Sonidos Personalizados!")
print("=" * 50)
print("Controles:")
print("  ✍️  1 dedo     - Dibujar")
print("  🖐️  2 dedos    - Cambiar color (🔊 con sonido)")
print("  ✋  4+ dedos   - Borrar (🔊 con sonido)")
print("  🤏  Pinza      - Mover trazos (🔊 con sonido)")
print("  ❌  Presiona 'q' para salir")
print("=" * 50)

while running:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        stop_drag()
        if current_stroke:
            missing_draw_frames += 1
            if missing_draw_frames > 6:
                commit_current_stroke()

    if results.multi_hand_landmarks:
        for index, handLms in enumerate(results.multi_hand_landmarks):
            hand_label = None
            if results.multi_handedness and index < len(results.multi_handedness):
                hand_label = results.multi_handedness[index].classification[0].label

            point = hand_point(handLms, img.shape)
            gesture = detect_gesture(handLms, img.shape, hand_label)

            handle_gesture(gesture, point)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    canvas = redraw_canvas(img.shape)

    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # Mostrar color actual
    cv2.rectangle(img, (10,10), (100,100), colors[color_index], -1)
    
    # Mostrar texto de instrucciones
    cv2.putText(img, "1 dedo dibuja | 2 dedos cambia color | palma borra | pinza mueve", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Mostrar indicador de sonidos personalizados
    if any(sonidos_existentes.values()):
        cv2.putText(img, "🔊 Sonidos Personalizados ACTIVADOS", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Air Drawing PRO", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n👋 Programa cerrado correctamente")