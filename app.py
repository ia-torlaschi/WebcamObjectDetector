from flask import Flask, render_template, Response, request, jsonify
from src.camera import Camera
import argparse
import os
import signal
import threading

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

# Global Camera Instance
camera = None

def generate_frames():
    while True:
        if camera:
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    task = data.get('task')
    conf = data.get('conf')
    model_base = data.get('model') # receiving base name now
    
    camera.set_settings(task=task, conf=conf, model_base=model_base)
    return jsonify({
        "status": "success", 
        "task": camera.task, 
        "conf": camera.conf, 
        "model": getattr(camera, 'model_path', 'unknown')
    })

@app.route('/shutdown', methods=['POST'])
def shutdown():
    print("Shutting down server...")
    if camera:
        camera.stop()
    
    # Schedule shutdown in a separate thread to allow response to return
    def kill_server():
        os.kill(os.getpid(), signal.SIGINT) # Robust kill for Flask dev server
    
    threading.Timer(0.5, kill_server).start()
    return jsonify({"status": "shutdown_initiated"})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="Camera Source")
    parser.add_argument("--device", default="cpu", help="Device (cpu/0)")
    args = parser.parse_args()

    # Initialize Camera
    try:
        camera = Camera(source=int(args.source) if str(args.source).isdigit() else args.source, 
                        device=args.device)
    except Exception as e:
        print(f"Error starting camera: {e}")
        exit(1)

    app.run(host='0.0.0.0', port=5000, debug=False)
