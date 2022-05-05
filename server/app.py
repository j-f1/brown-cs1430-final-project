from flask import Flask, request, make_response
import json
import base64

from faceswap import swap_face
from teeth import are_there_teeth
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, World!"


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


@app.route("/select-image", methods=["POST", "OPTIONS"])
def select_image():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    image = request.files["image"]
    data = image.read()
    image = cv2.cvtColor(np.array(Image.fromarray(data)), cv2.COLOR_BGR2GRAY)
    # face-detect
    faces = [{"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.1}]
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    json.dump(
        {
            "image": str(base64.encodebytes(data), encoding="ascii"),
            "faces": faces,
        },
        response.stream,
    )
    return response


@app.route("/swap", methods=["POST", "OPTIONS"])
def swap_faces():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    json.dump(
        {
            "image": "Working",
        },
        response.stream,
    )
    return response
