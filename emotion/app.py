import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
from deepface import DeepFace

# Configuración de la aplicación Flask
app = Flask(__name__)

# Configurar la carpeta de subida
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Máximo tamaño de archivo 16MB

# Asegúrate de que la carpeta de subidas exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Función para comprobar si el archivo tiene una extensión permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Inicializar MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

def analyze_face(image_path):
    try:
        # Leer la imagen
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("No se pudo cargar la imagen")

        # Convertir la imagen a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar puntos faciales
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No se detectó ninguna cara en la imagen")

        # Seleccionar los puntos clave (12 puntos principales de la cara)
        key_points = [70, 55, 285, 300, 33, 480, 133, 362, 473, 263, 4, 185, 0, 306, 1]
        height, width = gray_image.shape

        # Preparar transformaciones
        transformations = [
            ("Original", gray_image),
            ("Horizontally Flipped", cv2.flip(gray_image, 1)),
            ("Brightened", cv2.convertScaleAbs(gray_image, alpha=1.2, beta=50)),
            ("Upside Down", cv2.flip(gray_image, 0))
        ]

        # Inicializar la figura para mostrar
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for ax, (title, img) in zip(axes, transformations):
            ax.imshow(img, cmap='gray')
            num_landmarks = len(results.multi_face_landmarks[0].landmark)
            for point_idx in key_points:
                if point_idx < num_landmarks:
                    landmark = results.multi_face_landmarks[0].landmark[point_idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    # Ajustar puntos clave según las transformaciones
                    if title == "Horizontally Flipped":
                        x = width - x
                    elif title == "Upside Down":
                        y = height - y
                    ax.plot(x, y, 'rx')
            ax.set_title(title)
            ax.axis('off')

        # Guardar la imagen generada en memoria
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Convertir a base64 para mostrar en la web
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Análisis de emociones con DeepFace
        emotion = analyze_emotion(image_path)  # Usando DeepFace para analizar la emoción

        return image_base64, emotion

    except Exception as e:
        print(f"Error en analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

def analyze_emotion(image_path):
    # Usar DeepFace para analizar la emoción
    try:
        analysis = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception as e:
        print(f"Error en DeepFace: {str(e)}")
        return "Emotion detection failed"

@app.route('/')
def home():
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Revisar si estamos analizando un archivo existente
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': f'Archivo no encontrado: {filename}'}), 404

        # Revisar si estamos subiendo un archivo nuevo
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No se seleccionó un archivo'}), 400

            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no permitido'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        else:
            return jsonify({'error': 'No se proporcionó archivo'}), 400

        # Analizar la imagen
        result_image, emotion = analyze_face(filepath)

        return jsonify({
            'success': True,
            'image': result_image,
            'emotion': emotion
        })

    except Exception as e:
        print(f"Error en /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
