from flask import Flask, request, jsonify
from PIL import Image
import requests
import io as pyio
import numpy as np
from face_count import *
from numberprog import *
from facematch import *
from docmatch import *


app = Flask(__name__)


#Endpoint for face counting model

@app.route('/facecount', methods=['POST'])
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
    
    return jsonify({"prediction":prediction})

#Endpoint for phone number checking model

@app.route('/numbercheck', methods=['POST'])
def numbercheck():
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
    prediction = pocr(np.array(image))
    
    return jsonify({"prediction":prediction})



#Endpoint for face matching model

@app.route('/facematch', methods=['POST'])
def facematch():
    data = request.get_json()
    if 'image_url1' not in data and 'image_url2' not in data:
        return jsonify({"error": "No image URL provided"}), 400
    
    image_url1 = data['image_url1']
    image_url2 = data['image_url2']
    
    try:
        response1 = requests.get(image_url1)
        response2 = requests.get(image_url2)
        response1.raise_for_status()
        response2.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to fetch image from URL: {str(e)}"}), 400
    try:
        image1 = Image.open(BytesIO(response1.content))
        image2 = Image.open(BytesIO(response2.content))

    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400
    
    # Call your ML model here
    prediction = compare_faces(np.array(image1),np.array(image2))
    
    return jsonify({"prediction":prediction})

@app.route('/verify_aadhar', methods=['POST'])
def verify_aadhar():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json_data = request.get_json()
        img_url = json_data["image_url"]
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        face_count = check_if_aadhar(img)
        return jsonify({"doc_state":face_count})
    else:
        return "Content type is not supported."

@app.route('/verify_pan', methods=['POST'])
def verify_pan():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json_data = request.get_json()
        img_url = json_data["image_url"]
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        face_count = check_if_pan(img)
        return jsonify({"doc_state":face_count})
    else:
        return "Content type is not supported."
    
@app.route('/verify_dl', methods=['POST'])
def verify_dl():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json_data = request.get_json()
        img_url = json_data["image_url"]
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        face_count = check_if_dl(img)
        return jsonify({"doc_state":face_count})
    else:
        return "Content type is not supported."

if __name__ == '__main__':
    app.run(debug=True)