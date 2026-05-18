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

SONIDO_COLOR = "sonidos/cambio_color.wav"
SONIDO_BORRAR = "sonidos/borrar.wav"
SONIDO_ARRANQUE = "sonidos/arrastrar.wav"

def verificar_sonidos():
    sonidos_disponibles = {
        'color': os.path.exists(SONIDO_COLOR),
        'erase': os.path.exists(SONIDO_BORRAR),
        'drag': os.path.exists(SONIDO_ARRANQUE)
    }
    print("\n🔊 Verificando archivos de sonido:")
    for nombre, existe in sonidos_disponibles.items():
        print(f"  {'✅' if existe else '⚠️'} {nombre}: {'encontrado' if existe else 'NO encontrado - usando beep'}")
    return sonidos_disponibles

sonidos_existentes = verificar_sonidos()

def reproducir_sonido_accion(tipo):
    if tipo == 'color' and sonidos_existentes['color']:
        winsound.PlaySound(SONIDO_COLOR, winsound.SND_FILENAME | winsound.SND_ASYNC)
    elif tipo == 'erase' and sonidos_existentes['erase']:
        winsound.PlaySound(SONIDO_BORRAR, winsound.SND_FILENAME | winsound.SND_ASYNC)
    elif tipo == 'drag' and sonidos_existentes['drag']:
        winsound.PlaySound(SONIDO_ARRANQUE, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        if tipo == 'color':
            winsound.Beep(1046, 100)
        elif tipo == 'erase':
            winsound.Beep(330, 200)
        elif tipo == 'drag':
            winsound.Beep(660, 150)
        elif tipo == 'excelente':
            winsound.Beep(1046, 200)
        elif tipo == 'bien':
            winsound.Beep(880, 150)
        elif tipo == 'regular':
            winsound.Beep(660, 100)

# ============================================
# MODO DE JUEGO: FIGURAS PARA CALCAR
# ============================================

class ModoCalcar:
    def __init__(self):
        self.activo = False
        self.figura_actual = None
        self.nombre_figura = ""
        self.nivel = 0
        self.puntuacion = 0
        self.dibujo_usuario = []
        self.precision = 0
        self.esperando_dibujo = True
        
        # Figuras para calcar
        self.figuras = {
            'circulo': self.generar_circulo(),
            'cuadrado': self.generar_cuadrado(),
            'triangulo': self.generar_triangulo(),
            'linea': self.generar_linea(),
            'numero_1': self.generar_numero_1(),
            'numero_2': self.generar_numero_2()
        }
        
        self.niveles = [
            {'figura': 'linea', 'nombre': 'Línea Recta', 'puntos': 50},
            {'figura': 'circulo', 'nombre': 'Círculo', 'puntos': 100},
            {'figura': 'cuadrado', 'nombre': 'Cuadrado', 'puntos': 100},
            {'figura': 'triangulo', 'nombre': 'Triángulo', 'puntos': 100},
            {'figura': 'numero_1', 'nombre': 'Número 1', 'puntos': 150},
            {'figura': 'numero_2', 'nombre': 'Número 2', 'puntos': 150}
        ]
    
    def generar_circulo(self):
        puntos = []
        for angulo in range(0, 360, 15):
            rad = math.radians(angulo)
            x = 0.5 + 0.25 * math.cos(rad)
            y = 0.5 + 0.25 * math.sin(rad)
            puntos.append((x, y))
        return puntos
    
    def generar_cuadrado(self):
        return [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7), (0.3, 0.3)]
    
    def generar_triangulo(self):
        return [(0.5, 0.25), (0.75, 0.7), (0.25, 0.7), (0.5, 0.25)]
    
    def generar_linea(self):
        return [(0.2, 0.5), (0.8, 0.5)]
    
    def generar_numero_1(self):
        return [(0.5, 0.25), (0.5, 0.75)]
    
    def generar_numero_2(self):
        puntos = []
        for x in range(25, 76, 5):
            puntos.append((x/100, 0.3))
        for y in range(30, 71, 5):
            puntos.append((0.75 - (y-30)/100, y/100))
        for x in range(75, 24, -5):
            puntos.append((x/100, 0.7))
        return puntos
    
    def iniciar_juego(self):
        self.activo = True
        self.nivel = 0
        self.puntuacion = 0
        self.esperando_dibujo = True
        self.cargar_nivel()
        return True
    
    def cargar_nivel(self):
        if self.nivel < len(self.niveles):
            figura_info = self.niveles[self.nivel]
            self.figura_actual = self.figuras[figura_info['figura']]
            self.nombre_figura = figura_info['nombre']
            self.dibujo_usuario = []
            self.esperando_dibujo = True
            print(f"\n📐 Nivel {self.nivel + 1}: Dibuja un {self.nombre_figura}")
            return True
        else:
            self.activo = False
            return False
    
    def calcular_similitud(self, dibujo):
        if not self.figura_actual or len(dibujo) < 5:
            return 0
        
        def normalizar(puntos):
            if not puntos:
                return []
            xs = [p[0] for p in puntos]
            ys = [p[1] for p in puntos]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            ancho = max_x - min_x
            alto = max_y - min_y
            
            if ancho == 0:
                ancho = 1
            if alto == 0:
                alto = 1
            
            return [((x - min_x) / ancho, (y - min_y) / alto) for x, y in puntos]
        
        dibujo_norm = normalizar(dibujo)
        figura_norm = normalizar(self.figura_actual)
        
        num_puntos = min(50, len(dibujo_norm), len(figura_norm))
        paso_dibujo = max(1, len(dibujo_norm) // num_puntos)
        paso_figura = max(1, len(figura_norm) // num_puntos)
        
        dibujo_muestreado = dibujo_norm[::paso_dibujo][:num_puntos]
        figura_muestreada = figura_norm[::paso_figura][:num_puntos]
        
        distancias = []
        for i in range(min(len(dibujo_muestreado), len(figura_muestreada))):
            dx = dibujo_muestreado[i][0] - figura_muestreada[i][0]
            dy = dibujo_muestreado[i][1] - figura_muestreada[i][1]
            distancias.append(math.sqrt(dx*dx + dy*dy))
        
        if not distancias:
            return 0
        
        distancia_promedio = sum(distancias) / len(distancias)
        precision = max(0, min(100, 100 - (distancia_promedio * 100)))
        
        return precision
    
    def completar_dibujo(self, dibujo):
        if not dibujo or len(dibujo) < 5:
            return 0, 0
        
        precision = self.calcular_similitud(dibujo)
        puntos_nivel = self.niveles[self.nivel]['puntos']
        puntos_obtenidos = int(puntos_nivel * (precision / 100))
        self.puntuacion += puntos_obtenidos
        
        if precision > 80:
            print(f"  🌟 ¡Excelente! Precisión: {precision:.1f}% +{puntos_obtenidos} pts")
            reproducir_sonido_accion('excelente')
        elif precision > 60:
            print(f"  👍 ¡Bien! Precisión: {precision:.1f}% +{puntos_obtenidos} pts")
            reproducir_sonido_accion('bien')
        else:
            print(f"  📝 Regular. Precisión: {precision:.1f}% +{puntos_obtenidos} pts")
            reproducir_sonido_accion('regular')
        
        return precision, puntos_obtenidos
    
    def siguiente_nivel(self):
        self.nivel += 1
        if self.cargar_nivel():
            print(f"  ✨ ¡Siguiente nivel! Puntuación total: {self.puntuacion}")
            return True
        else:
            print(f"\n🎉 ¡FELICIDADES! Has completado todos los niveles!")
            print(f"🏆 Puntuación final: {self.puntuacion} puntos")
            return False
    
    def dibujar_figura_guia(self, img):
        if not self.activo or not self.figura_actual:
            return img
        
        h, w = img.shape[:2]
        
        for i, punto in enumerate(self.figura_actual):
            x = int(punto[0] * w)
            y = int(punto[1] * h)
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            
            if i > 0:
                x_ant = int(self.figura_actual[i-1][0] * w)
                y_ant = int(self.figura_actual[i-1][1] * h)
                cv2.line(img, (x_ant, y_ant), (x, y), (0, 255, 0), 3)
        
        nivel_actual = self.nivel + 1
        cv2.putText(img, f"Nivel: {nivel_actual}/{len(self.niveles)} - {self.nombre_figura}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Puntos: {self.puntuacion}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.esperando_dibujo:
            cv2.putText(img, "✍️ Dibuja la figura con 1 dedo", 
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return img

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
    raise RuntimeError("No se pudo abrir la camara")

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

modo_calcar = ModoCalcar()
modo_juego = False
mensaje_temporal = ""
tiempo_mensaje = 0

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
    global current_stroke, missing_draw_frames, modo_juego, mensaje_temporal, tiempo_mensaje
    
    if len(current_stroke) > 1:
        if modo_juego:
            # Convertir coordenadas a normalizadas
            h, w = img.shape[:2] if 'img' in dir() else (480, 640)
            dibujo_normalizado = [(x/w, y/h) for x, y in current_stroke]
            
            precision, puntos = modo_calcar.completar_dibujo(dibujo_normalizado)
            mensaje_temporal = f"Precisión: {precision:.1f}% +{puntos} pts"
            tiempo_mensaje = time.time()
            
            if not modo_calcar.siguiente_nivel():
                modo_juego = False
                mensaje_temporal = f"🎉 JUEGO COMPLETADO! Puntuación: {modo_calcar.puntuacion} 🎉"
                tiempo_mensaje = time.time()
        else:
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

def hand_point(handLms, img_shape):
    height, width, _ = img_shape
    index_tip = handLms.landmark[8]
    return int(index_tip.x * width), int(index_tip.y * height)

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
    finger_count = count_fingers(handLms, hand_label)
    if finger_count == 1:
        return "draw"
    if finger_count == 2:
        return "color"
    if finger_count >= 4:
        return "erase"
    return "idle"

def erase_nearest_stroke(point):
    if not strokes:
        return
    min_dist = float('inf')
    min_idx = -1
    for idx, stroke in enumerate(strokes):
        for px, py in stroke["points"]:
            dist = math.hypot(px - point[0], py - point[1])
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
    if min_idx != -1 and min_dist < 50:
        del strokes[min_idx]
        reproducir_sonido_accion('erase')

def handle_gesture(gesture, point):
    global current_stroke, missing_draw_frames, color_index, current_color, last_gesture
    
    if gesture == "draw":
        append_point_smoothly(current_stroke, point)
        missing_draw_frames = 0
        last_gesture = gesture
        return
    
    if current_stroke:
        missing_draw_frames += 1
        if missing_draw_frames > 6:
            commit_current_stroke()
    
    if gesture == "color" and last_gesture != "color":
        commit_current_stroke()
        color_index = (color_index + 1) % len(colors)
        current_color = colors[color_index]
        reproducir_sonido_accion('color')
    
    elif gesture == "erase" and last_gesture != "erase":
        commit_current_stroke()
        erase_nearest_stroke(point)
        reproducir_sonido_accion('erase')
    
    last_gesture = gesture

current_color = colors[color_index]
running = True
cv2.namedWindow("Air Drawing PRO")

print("=" * 60)
print("🎨 AIR DRAWING PRO - CON MODO JUEGO!")
print("=" * 60)
print("🎮 CONTROLES:")
print("   ✍️  1 dedo     - Dibujar")
print("   🖐️  2 dedos    - Cambiar color")
print("   ✋  4+ dedos   - Borrar")
print("")
print("🎮 MODO JUEGO (Calcar figuras):")
print("   🔘 Presiona 'G' o 'g' - Iniciar modo juego")
print("   🔘 Dibuja la figura que ves en VERDE")
print("   🔘 El programa evaluará tu dibujo")
print("   ❌ Presiona 'Q' o 'q' - Salir")
print("=" * 60)

while running:
    success, img = cap.read()
    if not success:
        continue
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for index, handLms in enumerate(results.multi_hand_landmarks):
            hand_label = None
            if results.multi_handedness and index < len(results.multi_handedness):
                hand_label = results.multi_handedness[index].classification[0].label
            
            point = hand_point(handLms, img.shape)
            gesture = detect_gesture(handLms, img.shape, hand_label)
            handle_gesture(gesture, point)
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
    else:
        if current_stroke:
            missing_draw_frames += 1
            if missing_draw_frames > 6:
                commit_current_stroke()
    
    canvas = redraw_canvas(img.shape)
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)
    
    # Modo juego
    if modo_juego:
        img = modo_calcar.dibujar_figura_guia(img)
        cv2.putText(img, "🎮 MODO CALCAR - Presiona 'G' para salir", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(img, "🎨 MODO DIBUJO - Presiona 'G' para jugar", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(img, (10,10), (100,100), colors[color_index], -1)
        cv2.putText(img, "1 dedo dibuja | 2 dedos color | 4+ dedos borrar", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Mostrar mensaje temporal
    if mensaje_temporal and time.time() - tiempo_mensaje < 3:
        cv2.putText(img, mensaje_temporal, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Mostrar estado de la tecla (solo para debug)
    cv2.putText(img, "Presiona 'G' para cambiar modo", (10, img.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    cv2.imshow("Air Drawing PRO", img)
    
    # DETECCIÓN DE TECLAS MEJORADA
    key = cv2.waitKey(1) & 0xFF
    
    # Mostrar en consola qué tecla se presiona (para debug)
    if key != 255:
        print(f"Tecla presionada: {chr(key) if 32 <= key <= 126 else key} (código: {key})")
    
    if key == ord('q') or key == ord('Q'):
        running = False
        print("Saliendo del programa...")
    
    elif key == ord('g') or key == ord('G'):
        print(f"\n🔘 Tecla G presionada! Modo juego actual: {modo_juego}")
        if not modo_juego:
            modo_juego = modo_calcar.iniciar_juego()
            strokes.clear()
            current_stroke.clear()
            print("🎮 Modo juego ACTIVADO!")
            print(f"   Nivel 1: Dibuja un {modo_calcar.niveles[0]['nombre']}")
            mensaje_temporal = "🎮 MODO JUEGO ACTIVADO!"
            tiempo_mensaje = time.time()
        else:
            modo_juego = False
            strokes.clear()
            current_stroke.clear()
            print("🎨 Modo dibujo libre ACTIVADO")
            mensaje_temporal = "🎨 MODO DIBUJO LIBRE"
            tiempo_mensaje = time.time()

cap.release()
cv2.destroyAllWindows()
print("\n👋 Programa cerrado correctamente")