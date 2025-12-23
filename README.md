# Webcam Object Detector 🎥

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO11](https://img.shields.io/badge/YOLO-v11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Activo-success)

**Una aplicación profesional de visión artificial en tiempo real para Object Detection, Instance Segmentation y Pose Estimation.**

Desarrollado por **[Torlaschi Consulting](https://github.com/ia-torlaschi)**.

---

## 🚀 Resumen

Este proyecto aprovecha la potencia de **YOLO11** (You Only Look Once) para ofrecer un análisis de alto rendimiento sobre streams de video o feeds de webcam. Está diseñado para ser **modular**, **eficiente** y **hardware-aware**, utilizando GPUs NVIDIA (CUDA) cuando están disponibles y haciendo fallback a CPU automáticamente cuando es necesario.

### Características Clave
- **🕵️ Object Detection**: Identificá y localizá objetos con bounding boxes y puntajes de confianza (confidence scores).
- **✂️ Segmentation**: Generá máscaras (masks) pixel-perfect para los objetos detectados.
- **🤸 Pose Estimation**: Trackeá keypoints esqueléticos humanos en tiempo real.
- **⚡ GPU/CPU Portable**: Autodetecta aceleración por hardware. Optimizado para NVIDIA RTX Series.
- **🔧 Control por CLI**: Totalmente configurable mediante argumentos de línea de comandos.

## 🛠️ Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/ia-torlaschi/WebcamObjectDetector.git
cd WebcamObjectDetector
```

### 2. Configurar el Entorno
Se recomienda usar un entorno virtual (virtual environment).

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
Este proyecto está optimizado para CUDA 12.4. El siguiente comando garantiza la versión correcta de PyTorch para aceleración por GPU:

```bash
pip install -r requirements.txt
```

### 4. Descargar Modelos
La aplicación intentará descargar los modelos automáticamente en la primera ejecución. Para configuración manual, verificá que tenés los siguientes archivos en el directorio raíz:
- `yolo11n.pt` (Detection)
- `yolo11n-seg.pt` (Segmentation)
- `yolo11n-pose.pt` (Pose Estimation)

> **Recomendación**: Usá los modelos `n` (Nano) para CPU o GPUs básicas. Usá `s` (Small) o `m` (Medium) para GPUs de gama alta.

## 💻 Uso

Corré la aplicación principal usando `python main.py`.

### 🔲 Object Detection
Modo de detección estándar (Bounding Boxes).
```bash
python main.py --task detect --model yolo11n.pt
```

### 🎭 Segmentation
Modo de segmentación de instancias (Masks + Boxes).
```bash
python main.py --task segment --model yolo11n-seg.pt
```

### 🦴 Pose Estimation
Modo de estimación de pose humana (Skeletons).
```bash
python main.py --task pose --model yolo11n-pose.pt
```

---

## ⚙️ Opciones de Configuración

| Argumento | Default | Descripción |
| :--- | :--- | :--- |
| `--model` | `yolo11n.pt` | Path al archivo del modelo YOLO (.pt). |
| `--task` | `detect` | Modo de ejecución: `detect`, `segment`, `pose`. |
| `--source` | `0` | Fuente de entrada. `0` para webcam default, `1` para externa, o path a un archivo de video. |
| `--conf` | `0.5` | Umbral de confianza (0.0 - 1.0). Filtra detecciones de baja confianza. |
| `--device` | `cpu` | Dispositivo de hardware. Usá `0` para GPU o `cpu` para procesador. |

**Ejemplo: Corriendo en GPU con alta confianza**
```bash
python main.py --task detect --device 0 --conf 0.70
```

## 🏗️ Estructura del Proyecto

```text
WebcamObjectDetector/
├── src/
│   ├── detector.py      # Lógica YOLO y gestión de hardware
│   ├── visualizer.py    # Utilidades de dibujado (Pillow/OpenCV)
│   └── utils.py         # Funciones auxiliares (Fonts, IO)
├── main.py              # Punto de entrada CLI
├── requirements.txt     # Dependencias (Congeladas con soporte CUDA)
└── README.md            # Documentación
```

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor abrí un issue o enviá un pull request para cualquier mejora o corrección de bugs.

---

## 👨‍💻 Autor

**Jorge Torlaschi**  
*Torlaschi Consulting*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jorge-torlaschi/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ia-torlaschi)
[![Website](https://img.shields.io/badge/Website-TorlaschiConsulting-blue?style=for-the-badge)](https://torlaschiconsulting.com/)

---
*Potenciando soluciones con Inteligencia Artificial.*