from flask import Flask, render_template, Response
from camera import VideoCamera

# Creamos la instancia de la aplicación Flask
app = Flask(__name__)

# Definir la ruta del servidor
@app.route('/')
def index():
    return render_template('index.html')

# Generar el flujo de video, definir la ruta para transmitirlo e iniciar el servidor web (local)
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
