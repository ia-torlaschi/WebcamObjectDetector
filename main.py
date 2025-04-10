import cv2
from ultralytics import YOLO
import math # Necesario para redondear la confianza

# Cargar el modelo YOLOv8 pre-entrenado
# Puedes probar 'yolov8n.pt' (más rápido, menos preciso) o 'yolov8s.pt' (balanceado)
# El modelo se descargará automáticamente si no existe localmente.
try:
    model = YOLO('yolov8s.pt')
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Obtener los nombres de las clases del modelo
class_names = model.names
# print("Clases detectables:", class_names) # Descomenta si quieres ver todas las clases

# Iniciar la captura de video desde la webcam (usualmente el índice 0)
cap = cv2.VideoCapture(0)

# Verificar si la webcam se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara web.")
    exit()

print("Presiona 'q' para salir...")

# Bucle principal para procesar cada frame del video
while True:
    # Leer un frame de la cámara
    success, frame = cap.read()

    # Si el frame se leyó correctamente
    if success:
        # Realizar la inferencia YOLO en el frame actual
        # stream=True es más eficiente para video
        try:
            results = model(frame, stream=True, verbose=False) # verbose=False para menos mensajes en consola
        except Exception as e:
            print(f"Error durante la inferencia: {e}")
            continue # Saltar al siguiente frame

        # Iterar sobre los resultados (detecciones) en el frame
        for r in results:
            boxes = r.boxes # Obtener el objeto Boxes con las detecciones

            # Iterar sobre cada cuadro delimitador detectado
            if boxes: # Comprobar si hay detecciones
                for box in boxes:
                    try:
                        # --- MÉTODO RECOMENDADO PARA EXTRAER DATOS ---
                        # 1. Obtener coordenadas del cuadro delimitador como lista de enteros
                        # box.xyxy es un tensor (ej. tensor([[x1, y1, x2, y2]])),
                        # accedemos a la primera fila  y convertimos a lista de enteros
                        xyxy = box.xyxy.int().tolist() # [[x1, y1, x2, y2]] -> [x1, y1, x2, y2]
                        x1, y1, x2, y2 = xyxy # Desempaquetar la lista

                        # 2. Obtener la confianza de la detección
                        # box.conf es un tensor (ej. tensor([confianza]))
                        # Usamos.item() para obtener el valor escalar de Python
                        confidence = math.ceil((box.conf.item() * 100)) / 100

                        # 3. Obtener el ID de la clase detectada
                        # box.cls es un tensor (ej. tensor([id_clase]))
                        # Usamos.item() para obtener el valor escalar y luego convertimos a int
                        cls_id = int(box.cls.item())
                        # --- FIN MÉTODO RECOMENDADO ---

                        # 4. Obtener el nombre de la clase
                        class_name = class_names[cls_id]

                        # 5. (Opcional) Filtrar detecciones con baja confianza
                        if confidence > 0.5: # Puedes ajustar este umbral (0.0 a 1.0)
                            # 6. Dibujar el cuadro delimitador en el frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Verde, grosor 2

                            # 7. Crear el texto del subtítulo (Nombre Clase: Confianza)
                            label = f'{class_name}: {confidence:.2f}'

                            # 8. Calcular tamaño del texto para el fondo
                            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                            # 9. Dibujar un fondo rectangular para el texto (con ajuste para borde superior)
                            text_bg_y1 = max(0, y1 - label_height - 5) # Evita que el fondo salga por arriba
                            cv2.rectangle(frame, (x1, text_bg_y1), (x1 + label_width, y1), (0, 255, 0), cv2.FILLED)

                            # 10. Dibujar el texto del subtítulo sobre el fondo (con ajuste para borde superior)
                            text_y = max(label_height + 5, y1 - 5) # Evita que el texto salga por arriba
                            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Negro

                    except IndexError:
                        print("Advertencia: Se encontró un tensor 'box.xyxy' con formato inesperado.")
                        continue # Saltar esta caja si el formato no es el esperado
                    except Exception as e:
                        print(f"Error procesando una caja: {e}")
                        continue # Saltar esta caja en caso de otro error

        # Mostrar el frame procesado en una ventana
        cv2.imshow('Detector de Objetos Webcam - YOLOv8', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Si no se pudo leer el frame (ej. fin del video o error de cámara)
    else:
        print("Error al leer el frame de la cámara.")
        break

# Liberar el objeto de captura de video
cap.release()
# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()

print("Aplicación cerrada.")