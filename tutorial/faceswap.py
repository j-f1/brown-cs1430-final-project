import cv2
# from dlib import get_frontal_face_detector, shape_predictor 
import dlib
# from scipy.spatial import Delaunay
# import numpy as np

#links to useful articles 
# https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c
# https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/
# https://www.cvlib.net
# 

#change path, store data in another folder
#will need to loop through all images eventually  

#this is the randomly generated face that will be placed on the head to anonomyize the image
src_img_path = "jennifer_aniston.png"
src_img = cv2.imread(src_img_path, 1)
bw_src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

#this is the image that the head will be placed on to 
dest_img_path = "jlo.png"
dest_img = cv2.imread(dest_img_path, 1)
bw_dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2GRAY)

#use python dlib to get the face
#get frontal_face_detector uses a HOG + Linear SVM face detection method which is faster than the CNN alternative
#an alternative option would be to use a CNN face detector which I believe is able to better identify faces in different angles, lightings, etc 
face_detector = dlib.get_frontal_face_detector()

#can use any shape predictor or could also train our own 
#downloaded the shape predictor 68 face landmark model which returns 68 key points in the face 
#this model came from this link: https://github.com/davisking/dlib-models
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_landmarks = dlib.shape_predictor(predictor_path)

#loop through all the faces returned from the front face detector in both the src and dest images
#could totally combine these two into a method and will do that later 
src_faces = face_detector(bw_src_img)
for face in src_faces:
    src_face_landmarks = face_landmarks(bw_src_img, face)
    # num_landmarks = src_face_landmarks.num_parts
    # src_face_landmark_points = np.empty((num_landmarks, 0))
    src_face_landmark_points = []
    
    for points in range(src_face_landmarks.num_parts):
        #add x and y coordinates of every point idenitfied by the predictor 
        # np.append(src_face_landmark_points, (src_face_landmarks.part(points).x, src_face_landmarks.part(points).y))
        src_face_landmark_points.append(src_face_landmarks.part(points).x)
        src_face_landmark_points.append(src_face_landmarks.part(points).y)
    
        # src_face_landmark_points = np.array(src_face_landmark_points)
        # src_face_landmark_points = np.reshape(src_face_landmark_points, (num_landmarks, 2))
        #testing 
#         cv2.circle(src_img, (src_face_landmarks.part(points).x, src_face_landmarks.part(points).y), 2, 255, 1)
#         cv2.putText(src_img, str(points), (src_face_landmarks.part(points).x + 4, src_face_landmarks.part(points).y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))

# cv2.imshow("window", src_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
  

dest_faces = face_detector(bw_dest_img)
for face in dest_faces:
    dest_face_landmarks = face_landmarks(bw_dest_img, face)
    num_landmarks = dest_face_landmarks.num_parts
    dest_face_landmark_points = []

    for points in range(dest_face_landmarks.num_parts):
        dest_face_landmark_points.append(dest_face_landmarks.part(points).x)
        dest_face_landmark_points.append(dest_face_landmarks.part(points).y)

    # dest_face_landmark_points = np.array(dest_face_landmark_points)
    # dest_face_landmark_points = np.reshape(dest_face_landmark_points, (num_landmarks, 2))

        # #testing 
        # cv2.circle(dest_img, (dest_face_landmarks.part(points).x, dest_face_landmarks.part(points).y), 2, 255, 1)
        # cv2.putText(dest_img, str(points), (dest_face_landmarks.part(points).x + 4, dest_face_landmarks.part(points).y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))

# cv2.imshow("window", dest_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Delaunay triangulation 





