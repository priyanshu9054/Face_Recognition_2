# # import cv2
# # import os
# # # Adjust the paths if necessary
# # cascade_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\haarcascade-frontalface-default.xml'
# # image_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\2.jpg'

# # # Check if files exist
# # if not os.path.isfile(cascade_path):
# #     print(f"Error: Haar cascade file '{cascade_path}' not found.")
# #     exit(1)

# # if not os.path.isfile(image_path):
# #     print(f"Error: Image file '{image_path}' not found.")
# #     exit(1)

# # # Load the Haar cascade file
# # face_cascade = cv2.CascadeClassifier(cascade_path)

# # # Read the input image
# # image = cv2.imread(image_path)

# # # Check if the image was successfully loaded
# # if image is None:
# #     print(f"Error: Unable to read image file '{image_path}'.")
# #     exit(1)

# # # Convert the image to grayscale
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # # Detect faces in the image
# # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # # Loop over the faces and extract them
# # for (x, y, w, h) in faces:
# #     # Crop the face
# #     cropped_face = image[y:y+h, x:x+w]
    
# #     # Save the cropped face
# #     face_filename = f"face_{x}_{y}.jpg"
# #     cv2.imwrite(face_filename, cropped_face)
# #     print(f"Cropped face saved as {face_filename}")

# # # Display the output
# # for (x, y, w, h) in faces:
# #     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# # cv2.imshow('Image with Detected Faces', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # import cv2
# # import numpy as np

# # # Function to rotate an image by 90 degrees clockwise
# # def rotate_image(image):
# #     return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# # # Load the input image
# # image_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\2.jpg'
# # image = cv2.imread(image_path)

# # # Check if the image was successfully loaded
# # if image is None:
# #     print(f"Error: Unable to read image file '{image_path}'.")
# #     exit(1)

# # # Initialize variables
# # face_found = False
# # rotation_attempts = 0

# # # Loop until a face is found or maximum rotations (3 rotations of 90 degrees) are reached
# # while not face_found and rotation_attempts < 3:
# #     # Convert the image to grayscale
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #     # Use OpenCV's built-in face detector (Haar Cascade)
# #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     # Check if any faces were found
# #     if len(faces) > 0:
# #         face_found = True
# #         break
    
# #     # Rotate the image by 90 degrees clockwise
# #     image = rotate_image(image)
# #     rotation_attempts += 1

# # # If a face is found, process it
# # if face_found:
# #     for (x, y, w, h) in faces:
# #         # Crop the face region from the image
# #         cropped_face = image[y:y+h, x:x+w]

# #         # Save the cropped face (optional)
# #         face_filename = f"face_{x}_{y}.jpg"
# #         cv2.imwrite(face_filename, cropped_face)
# #         print(f"Cropped face saved as {face_filename}")

# #         # Draw a rectangle around the detected face on the original image
# #         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# #     # Display the output image with detected faces
# #     cv2.imshow('Image with Detected Faces', image)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()

# # else:
# #     print("No faces found after rotating the image.")

# # import cv2
# # import mediapipe as mp

# # # Load MediaPipe Face Detection model
# # mp_face_detection = mp.solutions.face_detection
# # mp_drawing = mp.solutions.drawing_utils
# # face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# # # Load the input image
# # image_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\2.jpg'
# # image = cv2.imread(image_path)

# # # Check if the image was successfully loaded
# # if image is None:
# #     print(f"Error: Unable to read image file '{image_path}'.")
# #     exit(1)

# # # Convert the image to RGB (MediaPipe requires RGB images)
# # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # # Perform face detection
# # results = face_detection.process(image_rgb)

# # Check if any faces were detected
# # if results.detections:
# #     for detection in results.detections:
# #         # Get the bounding box of the face
# #         bboxC = detection.location_data.relative_bounding_box
# #         ih, iw, _ = image.shape
# #         bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        
# #         # Crop the face region from the image
# #         x, y, w, h = bbox
# #         cropped_face = image[y:y+h, x:x+w]
        
# #         # Save the cropped face (optional)
# #         face_filename = f"face_{x}_{y}.jpg"
# #         cv2.imwrite(face_filename, cropped_face)
# #         print(f"Cropped face saved as {face_filename}")

# #         # Draw a rectangle around the detected face on the original image
# #         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# # # Display the output image with detected faces
# # cv2.imshow('Image with Detected Faces', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # # Clean up
# # face_detection.close()


# # import cv2
# # import sys
# # import numpy as np
# # import cv2
# # import matplotlib.pyplot as plt

# # # imagePath = sys.argv[1]

# # image = cv2.imread('2.jpg')
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # height, width, _ = image.shape
# # if height > width:
# #     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
# # faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# # faces = faceCascade.detectMultiScale(
# #     gray,
# #     scaleFactor=1.3,
# #     minNeighbors=3,
# #     minSize=(30, 30)
# # )

# # print("[INFO] Found {0} Faces.".format(len(faces)))

# # for (x, y, w, h) in faces:
# #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #     roi_color = image[y:y + h, x:x + w]
# #     print("[INFO] Object found. Saving locally.")
# #     cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
# #     plt.imshow(roi_color)
# #     plt.show()

# # status = cv2.imwrite('faces_detected.jpg', image)
# # print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

# # install and import above modules first
# import os
# import cv2
# import math
# import matplotlib.pyplot as pl
# import pandas as pd
# from PIL import Image
# import numpy as np
 
# # Detect face
# def face_detection(img):
#     faces = face_detector.detectMultiScale(img, 1.1, 4)
#     if (len(faces) <= 0):
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return img, img_gray
#     else:
#         X, Y, W, H = faces[0]
#         img = img[int(Y):int(Y+H), int(X):int(X+W)]
#         return img, cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
 
 
# def trignometry_for_distance(a, b):
#     return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
#                      ((b[1] - a[1]) * (b[1] - a[1])))
 
# # Find eyes
# def Face_Alignment(img_path):
#     pl.imshow(cv2.imread(img_path)[:, :, ::-1])
#     pl.show()
#     img_raw = cv2.imread(img_path).copy()
#     img, gray_img = face_detection(cv2.imread(img_path))
#     eyes = eye_detector.detectMultiScale(gray_img)
 
#     # for multiple people in an image find the largest 
#     # pair of eyes
#     if len(eyes) >= 2:
#         eye = eyes[:, 2]
#         container1 = []
#         for i in range(0, len(eye)):
#             container = (eye[i], i)
#             container1.append(container)
#         df = pd.DataFrame(container1, columns=[
#                           "length", "idx"]).sort_values(by=['length'])
#         eyes = eyes[df.idx.values[0:2]]
 
#         # deciding to choose left and right eye
#         eye_1 = eyes[0]
#         eye_2 = eyes[1]
#         if eye_1[0] > eye_2[0]:
#             left_eye = eye_2
#             right_eye = eye_1
#         else:
#             left_eye = eye_1
#             right_eye = eye_2
 
#         # center of eyes
#         # center of right eye
#         right_eye_center = (
#             int(right_eye[0] + (right_eye[2]/2)), 
#           int(right_eye[1] + (right_eye[3]/2)))
#         right_eye_x = right_eye_center[0]
#         right_eye_y = right_eye_center[1]
#         cv2.circle(img, right_eye_center, 2, (255, 0, 0), 3)
 
#         # center of left eye
#         left_eye_center = (
#             int(left_eye[0] + (left_eye[2] / 2)), 
#           int(left_eye[1] + (left_eye[3] / 2)))
#         left_eye_x = left_eye_center[0]
#         left_eye_y = left_eye_center[1]
#         cv2.circle(img, left_eye_center, 2, (255, 0, 0), 3)
 
#         # finding rotation direction
#         if left_eye_y > right_eye_y:
#             print("Rotate image to clock direction")
#             point_3rd = (right_eye_x, left_eye_y)
#             direction = -1  # rotate image direction to clock
#         else:
#             print("Rotate to inverse clock direction")
#             point_3rd = (left_eye_x, right_eye_y)
#             direction = 1  # rotate inverse direction of clock
 
#         cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
#         a = trignometry_for_distance(left_eye_center, 
#                                      point_3rd)
#         b = trignometry_for_distance(right_eye_center, 
#                                      point_3rd)
#         c = trignometry_for_distance(right_eye_center, 
#                                      left_eye_center)
#         cos_a = (b*b + c*c - a*a)/(2*b*c)
#         angle = (np.arccos(cos_a) * 180) / math.pi
 
#         if direction == -1:
#             angle = 90 - angle
#         else:
#             angle = -(90-angle)
 
#         # rotate image
#         new_img = Image.fromarray(img_raw)
#         new_img = np.array(new_img.rotate(direction * angle))
 
#     return new_img
 
 
# opencv_home = cv2.__file__
# folders = opencv_home.split(os.path.sep)[0:-1]
# path = folders[0]
# for folder in folders[1:]:
#     path = path + "/" + folder
# path_for_face = path+"D:\Lenditt\Face_Recognoition\Detect_face\haarcascade-frontalface-default.xml"
# path_for_eyes = path+"D:\Lenditt\Face_Recognoition\Detect_face\haarcascade-eye.xml"
# path_for_nose = path+"D:\Lenditt\Face_Recognoition\Detect_face\haarcascade_mcs_nose.xml"
 
# if os.path.isfile(path_for_face) != True:
#     raise ValueError(
#         "opencv is not installed pls install using pip install opencv ", 
#         " violated.")
 
# face_detector = cv2.CascadeClassifier(path_for_face)
# eye_detector = cv2.CascadeClassifier(path_for_eyes)
# nose_detector = cv2.CascadeClassifier(path_for_nose)
 
# # Name of the image for face alignment if on 
# # the other folder kindly paste the name of
# # the image with path included
# test_set = ["2.jpg"]
# for i in test_set:
#     alignedFace = Face_Alignment(i)
#     pl.imshow(alignedFace[:, :, ::-1])
#     pl.show()
#     img, gray_img = face_detection(alignedFace)
#     pl.imshow(img[:, :, ::-1])
#     pl.show()