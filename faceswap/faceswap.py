import cv2
import numpy as np
from dlib import get_frontal_face_detector, shape_predictor
import time

TEETH_TRIANGLES = (111, 110)


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


#this is the randomly generated face that will be placed on the head to anonomyize the image
src_img_path = "emily.jpg"
src_img = cv2.imread(src_img_path, 1)
bw_src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

# this is the image that the head will be placed on to
dest_img_path = "jlo.png"
dest_img = cv2.imread(dest_img_path, 1)
bw_dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(bw_src_img)

# use python dlib to get the face
# get frontal_face_detector uses a HOG + Linear SVM face detection method which is faster than the CNN alternative
# an alternative option would be to use a CNN face detector which I believe is able to better identify faces in different angles, lightings, etc
face_detector = get_frontal_face_detector()

# can use any shape predictor or could also train our own
# downloaded the shape predictor 68 face landmark model which returns 68 key points in the face
# this model came from this link: https://github.com/davisking/dlib-models
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_landmarks_predictor = shape_predictor(shape_predictor_path)

new_face = np.zeros((dest_img.shape), np.uint8)

#Source image
faces = face_detector(bw_src_img)
for face in faces:
    landmarks = face_landmarks_predictor(bw_src_img, face)
    num_landmarks = landmarks.num_parts
    landmarks_points = []

    for points in range(num_landmarks):
        landmarks_points.append((landmarks.part(points).x, landmarks.part(points).y))

    points = np.array(landmarks_points)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)

    cv2.bitwise_and(src_img, src_img, mask=mask)

    # use Delaunay triangulation to get triangles in between each landmark point found
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = np.array(subdiv.getTriangleList())

    indexes_triangles = []

    for coord in triangles:
        point1 = (coord[0], coord[1])
        point2 = (coord[2], coord[3])
        point3 = (coord[4], coord[5])

        index1 = np.where((points == point1).all(axis=1))
        index1 = extract_index_nparray(index1)

        index2 = np.where((points == point2).all(axis=1))
        index2 = extract_index_nparray(index2)

        index3 = np.where((points == point3).all(axis=1))
        index3 = extract_index_nparray(index3)

        if index1 is not None and index2 is not None and index3 is not None:
            triangle = [index1, index2, index3]
            indexes_triangles.append(triangle)


#Destination image 
dest_faces = face_detector(bw_dest_img)
face_tris = []
for face in dest_faces:
    landmarks = face_landmarks_predictor(bw_dest_img, face)
    landmarks_points2 = []
    
    for points in range(num_landmarks):
        landmarks_points2.append((landmarks.part(points).x, landmarks.part(points).y))

    points2 = np.array(landmarks_points2, np.int32)
    convexhull2 = cv2.convexHull(points2)

    #Delaunay triangulation to get triangles in between each landmark point found
    rect2 = cv2.boundingRect(convexhull2)
    subdiv2 = cv2.Subdiv2D(rect2)
    subdiv2.insert(landmarks_points2)
    triangles2 = np.array(subdiv2.getTriangleList())

    areas = 0.5 * (
        triangles2[:, 0] * (triangles2[:, 3] - triangles2[:, 5])
        + triangles2[:, 2] * (triangles2[:, 5] - triangles2[:, 1])
        + triangles2[:, 4] * (triangles2[:, 1] - triangles2[:, 3])
    )
    triangle_xs = triangles2[:, ::2]
    triangle_ys = triangles2[:, 1::2]
    triangle_centers = (np.mean([triangle_xs, triangle_ys], axis=-1)).T
    face_tris.append(
        (triangle_centers, np.stack([[triangle_xs], [triangle_ys]], axis=-1), areas)
    )


lines_space_mask = np.zeros_like(bw_src_img)
lines_space_new_face = np.zeros_like(dest_img)

#triangulation of both the faces 
for triangle_index in indexes_triangles:
    
    #triangulation of the first face
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    # rect1 = cv2.boundingRect(triangle1)
    # (x, y, w, h) = rect1
    (x, y, w, h) = cv2.boundingRect(triangle1)
     
    cropped_triangle = src_img[y : y + h, x : x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)

    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],[tr1_pt2[0] - x, tr1_pt2[1] - y],[tr1_pt3[0] - x, tr1_pt3[1] - y],],np.int32,)

    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

    # Lines space
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
    cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
    lines_space = cv2.bitwise_and(src_img, src_img, mask=lines_space_mask)

    # Triangulation of second face
    tr2_pt1 = landmarks_points2[triangle_index[0]]
    tr2_pt2 = landmarks_points2[triangle_index[1]]
    tr2_pt3 = landmarks_points2[triangle_index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

    (x, y, w, h) = cv2.boundingRect(triangle2)
    

    cropped_tr2_mask = np.zeros((h, w), np.uint8)

    points2 = np.array(
        [
            [tr2_pt1[0] - x, tr2_pt1[1] - y],
            [tr2_pt2[0] - x, tr2_pt2[1] - y],
            [tr2_pt3[0] - x, tr2_pt3[1] - y],
        ],
        np.int32,
    )

    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

    # Warp triangles
    points = np.float32(points)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

    # Reconstructing destination face
    img2_new_face_rect_area = new_face[y : y + h, x : x + w]
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)

    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)

    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    new_face[y : y + h, x : x + w] = img2_new_face_rect_area


# Face swapped (putting 1st face into 2nd face)
img2_face_mask = np.zeros_like(bw_dest_img)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)

img2_head_noface = cv2.bitwise_and(dest_img, dest_img, mask=img2_face_mask)
result = cv2.add(img2_head_noface, new_face)

(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

seamlessclone = cv2.seamlessClone(
    result, dest_img, img2_head_mask, center_face2, cv2.NORMAL_CLONE
)


# for triangle_centers, points, areas in face_tris:
#     cv2.polylines(seamlessclone, np.int32(points[0]), True, (255, 0, 0))
#     print(areas[np.array(TEETH_TRIANGLES)])
#     for i, center in enumerate(triangle_centers):
#         cv2.putText(
#             seamlessclone,
#             str(i),
#             np.int32(center),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.4,
#             (255, 255, 255),
#         )


cv2.imshow("seamlessclone", seamlessclone)
cv2.waitKey(0)
cv2.destroyAllWindows()
