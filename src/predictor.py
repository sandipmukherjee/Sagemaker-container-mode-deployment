import base64
import io
import os

import flask
from flask import Flask, request, jsonify

from inference.classifier import DiSegmentPredictor

app = Flask(__name__)
global model
model = None
# app.model = load_model("s3://kittl/saved_model_dis/isnet.pth")

# @app.before_first_request
# def before_first_request_func():
#     MOODEL WEIGHT LOADING CODE
#     return model


# tells Flask what URL a user has to browse to call the function below.
# you will need to browse the url : '/test'
@app.route("/test")
def test():
    return "First app in Flask"


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In container, we declare
    it healthy if we can load the model successfully."""
    # health = DiSegmentPredictor.get_model() is not None  # we should insert a health check here

    status = 200 # if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route("/invocations", methods=["POST"])
def segment():
    """
    Initialize the model id not initialized
    :return:
    """
    global model
    if model is None:
        model_path = os.environ['MODEL_PATH']
        model = DiSegmentPredictor(os.path.join(model_path, 'isnet.pth'))
    data = request.json
    img_path = data.get('img_path')
    img = model.predict(img_path)
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')  # convert the PIL image to byte array
    encoded_img = base64.encodebytes(img_io.getvalue()).decode('ascii')
    return jsonify({'img': encoded_img})


if __name__ == "__main__":
    app.run(debug=True)