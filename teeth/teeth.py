# based on tutorial/tutorial.py

import cv2
import numpy as np
from dlib import get_frontal_face_detector, shape_predictor
import time

CUTOFF = 1

# clrs.cc
GREEN = (46, 204, 64)
BLUE = (0, 116, 217)
PURPLE = (177, 13, 201)


def dlib_point_to_tuple(pt):
    return (pt.x, pt.y)


shape_predictor_path = "../faceswap/shape_predictor_68_face_landmarks.dat"
face_landmarks_predictor = shape_predictor(shape_predictor_path)
face_detector = get_frontal_face_detector()


def are_there_teeth(img, annotate):
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(bw_img)
    if len(faces) == 0:
        if annotate:
            return img, "unknown"
        else:
            return "unknown"

    face = faces[0]
    mask = np.zeros_like(bw_img)
    if annotate:
        out_img = np.array(img)
        cv2.rectangle(
            out_img,
            dlib_point_to_tuple(face.tl_corner()),
            dlib_point_to_tuple(face.br_corner()),
            GREEN,
        )
    landmarks = face_landmarks_predictor(bw_img, face)

    landmark_verts = np.array(
        [dlib_point_to_tuple(landmarks.part(i)) for i in range(landmarks.num_parts)],
        dtype=np.int32,
    )

    if annotate:
        for vert in landmark_verts:
            cv2.circle(out_img, vert, 2, BLUE)

    convexhull = cv2.convexHull(landmark_verts)
    cv2.fillConvexPoly(mask, convexhull, 255)
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_verts.astype(float))
    triangles = np.array(subdiv.getTriangleList())
    areas = 0.5 * (
        triangles[:, 0] * (triangles[:, 3] - triangles[:, 5])
        + triangles[:, 2] * (triangles[:, 5] - triangles[:, 1])
        + triangles[:, 4] * (triangles[:, 1] - triangles[:, 3])
    )

    heuristic = areas[np.array(-2)] / face.area() * 1e3

    if annotate:
        return out_img, (":D" if heuristic > CUTOFF else ":|")
    return heuristic


if __name__ == "__main__":
    vid = cv2.VideoCapture(0)

    while True:
        ok, img = vid.read()
        print(np.max(img))
        if not ok:
            break

        out_img, status = are_there_teeth(img, annotate=True)
        cv2.putText(out_img, status, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, PURPLE)
        cv2.imshow("frame", out_img)

        if cv2.waitKey(1) == 0o33:
            # esc
            break

    cv2.destroyAllWindows()
    vid.release()
