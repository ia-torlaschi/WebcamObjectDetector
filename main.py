import cv2
from ultralytics import YOLO
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Importaciones de Pillow
import os

# --- Configuración ---
MODEL_NAME = 'yolo11n.pt'
CONFIDENCE_THRESHOLD = 0.5
WEBCAM_INDEX = 0

# --- Configuración de Apariencia del Texto y Cajas ---
FONT_FILENAME = "DejaVuSans.ttf" # ¡Verifica que existe! O usa "arial.ttf", etc.
FONT_SIZE = 18
TEXT_PADDING = 4           # Píxeles de padding alrededor del texto
GAP_BELOW_BOX = 5          # Espacio vertical entre caja de objeto y fondo de texto
BACKGROUND_COLOR = (0, 255, 0) # Fondo del texto
TEXT_COLOR = (0, 0, 0)       # Color del texto
BBOX_COLOR = (0, 255, 0)     # Color del cuadro del objeto
BBOX_THICKNESS = 2         # Grosor del cuadro del objeto
# --- Fin Configuración ---

# --- Diccionario de Traducción ELIMINADO ---

# --- Cargar Fuente ---
try:
    font = ImageFont.truetype(FONT_FILENAME, FONT_SIZE)
    print(f"Fuente '{FONT_FILENAME}' cargada correctamente (tamaño {FONT_SIZE}).")
except IOError:
    print(f"Advertencia: No se pudo cargar la fuente TTF '{FONT_FILENAME}'.")
    print("Intentando cargar fuente por defecto.")
    try:
        font = ImageFont.load_default(FONT_SIZE)
    except AttributeError:
        font = ImageFont.load_default()
# --- Fin Carga Fuente ---

# --- Carga del Modelo YOLO ---
try:
    print(f"Cargando modelo {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    # Obtenemos los nombres originales en inglés
    class_names_en = model.names
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error crítico al cargar el modelo YOLO: {e}")
    exit()
# --- Fin Carga Modelo ---

# --- Iniciar Webcam ---
print(f"Iniciando captura de webcam (índice {WEBCAM_INDEX})...")
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error crítico: No se pudo abrir la cámara web con índice {WEBCAM_INDEX}.")
    exit()
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam abierta ({frame_width}x{frame_height}). Presiona 'q' para salir.")
# --- Fin Iniciar Webcam ---

# === Bucle Principal ===
while True:
    # 1. Leer Frame (OpenCV)
    success, frame = cap.read()
    if not success:
        print("Error al leer el frame de la cámara o fin del stream.")
        break

    # 2. Inferencia YOLO
    try:
        results = model(frame, stream=True, verbose=False, device=0)
    except Exception as e:
        print(f"Error durante la inferencia YOLO: {e}")
        continue

    # 3. Preparar Lienzo Pillow (Convertir a Pillow RGB)
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
    except Exception as e:
        print(f"Error al convertir frame a imagen Pillow: {e}")
        continue

    # 4. Procesar Resultados y Dibujar SOBRE LA IMAGEN PILLOW
    for r in results:
        boxes = r.boxes
        if boxes:
            for box in boxes:
                try:
                    # a. Extraer datos de la caja
                    coords = box.xyxy.int().tolist()[0]
                    x1, y1, x2, y2 = coords
                    confidence = math.ceil((box.conf.item() * 100)) / 100
                    cls_id = int(box.cls.item())

                    # b. Obtener nombre en inglés y filtrar por confianza
                    # Ya NO traducimos
                    class_name_en = class_names_en[cls_id]

                    if confidence >= CONFIDENCE_THRESHOLD:

                        # --- Dibujo TODO con Pillow ---
                        try:
                            # c. Dibujar Bounding Box del objeto (Pillow)
                            draw.rectangle(
                                [(x1, y1), (x2, y2)],
                                outline=BBOX_COLOR,
                                width=BBOX_THICKNESS
                            )

                            # d. Preparar Etiqueta (Inglés)
                            label = f'{class_name_en}: {confidence:.2f}' # Usamos class_name_en

                            # e. Calcular tamaño necesario para el texto
                            text_bbox = draw.textbbox((0, 0), label, font=font)
                            label_width = text_bbox[2] - text_bbox[0]
                            label_height = text_bbox[3] - text_bbox[1]

                            # f. Calcular posición DEBAJO de la caja (LÓGICA REVISADA)
                            bg_y1 = y2 + GAP_BELOW_BOX # El fondo empieza 'gap' píxeles bajo la caja
                            text_y = bg_y1 + TEXT_PADDING # El texto empieza dentro del padding del fondo
                            text_x = x1 # Texto alineado a la izquierda

                            # Calcular altura total del fondo (texto + padding superior/inferior)
                            total_bg_height = label_height + (2 * TEXT_PADDING)
                            bg_y2 = bg_y1 + total_bg_height

                            # g. Comprobar si cabe verticalmente y dibujar
                            if bg_y2 < frame_height: # Comprobar borde inferior
                                # Calcular coordenadas X del fondo CON padding
                                bg_x1 = max(0, text_x - TEXT_PADDING)
                                bg_x2 = text_x + label_width + TEXT_PADDING

                                # Dibujar fondo (Pillow)
                                draw.rectangle(
                                    [(bg_x1, bg_y1), (bg_x2, bg_y2)],
                                    fill=BACKGROUND_COLOR
                                )
                                # Dibujar texto (Pillow)
                                draw.text((text_x, text_y), label, font=font, fill=TEXT_COLOR)

                        except Exception as e_draw:
                             print(f"Error dibujando con Pillow: {e_draw}")
                        # --- Fin Dibujo Pillow ---

                except Exception as e_box:
                     print(f"Error procesando una caja específica: {e_box}")
                     continue

    # 5. Finalizar Dibujo (Convertir de Pillow RGB a OpenCV BGR)
    try:
        frame_final = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e_conv:
        print(f"Error al convertir imagen Pillow a frame OpenCV: {e_conv}")
        frame_final = frame

    # 6. Mostrar Frame Final
    # Cambiamos el título de la ventana para reflejar que usa inglés
    cv2.imshow('Object Detector Webcam - YOLOv8 (English Labels)', frame_final)

    # 7. Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Saliendo por petición del usuario...")
        break
# === Fin Bucle Principal ===

# --- Liberar Recursos ---
print("Liberando recursos...")
cap.release()
cv2.destroyAllWindows()
print("Aplicación cerrada.")
# --- Fin Liberar Recursos ---