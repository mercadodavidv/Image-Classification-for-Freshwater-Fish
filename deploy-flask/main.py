# Install dependencies:
# $ pip install Flask==2.0.1 torchvision==0.10.0

import io
import json

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)
imagenet_class_index = json.load(open('./imagenet_class_index.json'))
model = torch.load('../model.pth')
model.to('cpu')
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


def get_prediction(image_bytes, max_results=2):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    #_, y_hat = outputs.max(1)
    #predicted_idx = str(y_hat.item())
    #return imagenet_class_index[predicted_idx]
    values, preds = torch.topk(outputs, max_results, dim=1)
    sm = nn.functional.softmax(values, dim=1)
    results = []
    percentages = []
    for pred, perc in zip(preds[0], sm[0]):
        percentages.append(perc)
        results.append(imagenet_class_index[str(pred.item())])
    return results, percentages


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        results, percentages = get_prediction(image_bytes=img_bytes)
        jslist = [
            {'class_id': results[0][0], 'percentage': '{:.2f}'.format(percentages[0] * 100)},
            {'class_id': results[1][0], 'percentage': '{:.2f}'.format(percentages[1] * 100)}
        ]
        return json.dumps(jslist)


if __name__ == '__main__':
    app.run()
