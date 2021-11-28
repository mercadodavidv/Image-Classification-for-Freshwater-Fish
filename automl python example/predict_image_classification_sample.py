### USAGE:
# $ python -c "from predict_image_classification_sample import *; predict_image_classification_sample(project='621480173088', endpoint_id='3989494378501505024', location='us-central1', filename='YOUR_IMAGE_FILE')"
###
# prerequisite, install gcloud library in a virtual environment:
# (windows), might need to run [Set-ExecutionPolicy Unrestricted -Scope Process] in PS
# pip install virtualenv
# virtualenv <your-env>
# <your-env>\Scripts\activate
# <your-env>\Scripts\pip.exe install google-cloud-aiplatform
# (mac/linux)
# pip install virtualenv
# virtualenv <your-env>
# source <your-env>/bin/activate
# <your-env>/bin/pip install google-cloud-aiplatform
###

# from https://github.com/googleapis/python-aiplatform/blob/HEAD/samples/snippets/prediction_service/predict_image_classification_sample.py

import base64

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict


def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.0,
        max_predictions=5, # does not behave as expected. currently returns all 19 results. will need to filter out top 5 in app
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/classification.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))
