from flask import Flask, request, make_response
import json
import base64

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
    # TODO: read into numpy array
    # TODO: face-detect
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

@app.route("/swap", methods=["POST"])
def swap_faces():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    json.dump(
        {
            "image": "Working",
        },
        response.stream,
    )
    return response
