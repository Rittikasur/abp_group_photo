from flask import Flask, request, jsonify
from PIL import Image
import io as pyio
import numpy as np
from model import *

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400
    
    image_url = data['image_url']
    
    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to fetch image from URL: {str(e)}"}), 400
    
    try:
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400
    
    # Call your ML model here
    prediction = check_face_count(np.array(image))
    
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)