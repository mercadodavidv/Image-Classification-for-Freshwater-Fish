# Install dependencies:
# $ pip install Flask==2.0.1 torchvision==0.10.0
# Run test flask server
# $ python main.py

import io
import json

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, redirect

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)

imagenet_class_index = json.load(open('./imagenet_class_index.json')) # classes in here need to be in alphabetical order
model = torch.load('./model.pth', map_location=torch.device('cpu'))
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes, num_results=5):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    values, preds = torch.topk(outputs, num_results, dim=1)
    confidence_level = nn.functional.softmax(values, dim=1)
    results = []
    percentages = []
    for pred, perc in zip(preds[0], confidence_level[0]):
        percentages.append(perc)
        results.append(imagenet_class_index[str(pred.item())])
    return results, percentages

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file in request body"}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format"}), 400

        img_bytes = file.read()
        results, percentages = get_prediction(image_bytes=img_bytes, num_results=5)
        jslist = [
            {'class_id': results[0][0], 'percentage': round(float(percentages[0] * 100), 2)},
            {'class_id': results[1][0], 'percentage': round(float(percentages[1] * 100), 2)},
            {'class_id': results[2][0], 'percentage': round(float(percentages[2] * 100), 2)},
            {'class_id': results[3][0], 'percentage': round(float(percentages[3] * 100), 2)},
            {'class_id': results[4][0], 'percentage': round(float(percentages[4] * 100), 2)}
        ]
        return json.dumps(jslist)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
