from flask import Flask, request, make_response
import json
import base64
import os
import dlib
import random

import faceswap
from teeth import are_there_teeth, CUTOFF
from cnn_code.predict import predict_image
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

from matching import find_matches

app = Flask(__name__)


def dlib_rect_to_dict(rect, size, teeth):
    return {
        "x": rect.left() / size[1],
        "y": rect.top() / size[0],
        "width": rect.width() / size[1],
        "height": rect.height() / size[0],
        "teeth_heuristic": float(teeth),
        "teeth": bool(teeth > CUTOFF),
    }


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
    img = np.array(Image.open(image).convert("RGB"))
    sex = predict_image(img)

    # face-detect
    faces, heuristics = are_there_teeth(img, annotate=False)
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    json.dump(
        {
            "image": str(base64.encodebytes(data), encoding="ascii"),
            "faces": [
                dlib_rect_to_dict(rect, img.shape, teeth)
                for rect, teeth in zip(faces, heuristics)
            ],
            "sex": sex,
        },
        response.stream,
    )
    return response


@app.route("/swap", methods=["POST", "OPTIONS"])
def swap_faces():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    data = base64.b64decode(bytes(request.form["image"], encoding="ascii"))
    img = np.array(Image.open(BytesIO(data)).convert("RGB"))

    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")

    faces = find_matches(sex=request.form["sex"], teeth=request.form["teeth"])

    face_id = random.choice(faces)

    result = faceswap.swap_face(
        os.path.join(os.path.dirname(__file__), "../ai_faces/" + face_id + ".png"),
        img,
        dlib.rectangle(
            int(float(request.form["x"]) * img.shape[1]),
            int(float(request.form["y"]) * img.shape[0]),
            int(
                (float(request.form["width"]) + float(request.form["x"])) * img.shape[1]
            ),
            int(
                (float(request.form["height"]) + float(request.form["y"]))
                * img.shape[0]
            ),
        ),
    )

    json.dump(
        {
            "image": str(
                base64.encodebytes(cv2.imencode(".png", result)[1]),
                encoding="ascii",
            ),
        },
        response.stream,
    )
    return response
