# import cv2

# # Load an image
# image_path = '2.jpg'  # Replace with your image path
# image = cv2.imread(image_path)

# # Check if the image was successfully loaded
# if image is None:
#     print(f"Error: Unable to load image '{image_path}'")
# else:
#     h,w,n = image.shape 
#     print(h,w)
#     if h>w:
#         image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#     h,w,n = image.shape 

#     print(h,w)
#     # Display the image
#     # cv2.imshow('Image', image)

#     # # Wait for a key press and close all OpenCV windows
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
# import cv2
# import os
# # Adjust the paths if necessary
# cascade_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\haarcascade-frontalface-default.xml'
# image_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\2.jpg'

# # Check if files exist
# if not os.path.isfile(cascade_path):
#     print(f"Error: Haar cascade file '{cascade_path}' not found.")
#     exit(1)

# if not os.path.isfile(image_path):
#     print(f"Error: Image file '{image_path}' not found.")
#     exit(1)

# # Load the Haar cascade file
# face_cascade = cv2.CascadeClassifier(cascade_path)

# # Read the input image
# image = cv2.imread(image_path)

# # Check if the image was successfully loaded
# if image is None:
#     print(f"Error: Unable to read image file '{image_path}'.")
#     exit(1)

# h,w,n = image.shape 
# print(h,w)
# if h>w:
#     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# h,w,n = image.shape 

# print(h,w)
# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detect faces in the image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# print(len(faces))
# if len(faces)==0:
#     image = cv2.rotate(image, cv2.ROTATE_180)
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# print(len(faces))
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # Loop over the faces and extract them
# for (x, y, w, h) in faces:
#     # Crop the face
#     cropped_face = image[y:y+h, x:x+w]
    
    
#     # Save the cropped face
#     face_filename = f"face_{x}_{y}.jpg"
#     cv2.imwrite(face_filename, cropped_face)
#     print(f"Cropped face saved as {face_filename}")

# # Display the output
# # for (x, y, w, h) in faces:
# #     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# # cv2.imshow('Image with Detected Faces', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# import cv2
# import os

# # Paths to cascade and image
# cascade_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\haarcascade-frontalface-default.xml'
# image_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\5.jpg'

# # Check if files exist
# if not os.path.isfile(cascade_path):
#     print(f"Error: Haar cascade file '{cascade_path}' not found.")
#     exit(1)

# if not os.path.isfile(image_path):
#     print(f"Error: Image file '{image_path}' not found.")
#     exit(1)

# # Load the Haar cascade file
# face_cascade = cv2.CascadeClassifier(cascade_path)

# # Read the input image
# image = cv2.imread(image_path)

# # Check if the image was successfully loaded
# if image is None:
#     print(f"Error: Unable to read image file '{image_path}'.")
#     exit(1)

# # Check image dimensions and rotate if necessary
# h, w, _ = image.shape
# if h > w:
#     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detect faces in the grayscale image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30))

# if len(faces)==0:
#     image = cv2.rotate(image, cv2.ROTATE_180)

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detect faces in the grayscale image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# len(faces)
# # Draw rectangles around the detected faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
# cropped_face = image[y:y+h, x:x+w]

# # Save the cropped face
# face_filename = f"face_{x}_{y}_5.jpg"
# cv2.imwrite(face_filename, cropped_face)
# print(f"Cropped face saved as {face_filename}")
# # Display the output image with detected faces
# cv2.imshow('Image with Detected Faces', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import os

# # Paths to cascade and image
# cascade_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\haarcascade-frontalface-default.xml'
# image_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\5.jpg'

# # Check if files exist
# if not os.path.isfile(cascade_path):
#     print(f"Error: Haar cascade file '{cascade_path}' not found.")
#     exit(1)

# if not os.path.isfile(image_path):
#     print(f"Error: Image file '{image_path}' not found.")
#     exit(1)

# # Load the Haar cascade file
# face_cascade = cv2.CascadeClassifier(cascade_path)

# # Read the input image
# image = cv2.imread(image_path)

# # Check if the image was successfully loaded
# if image is None:
#     print(f"Error: Unable to read image file '{image_path}'.")
#     exit(1)

# # Check image dimensions and rotate if necessary
# h, w, _ = image.shape
# if h > w:
#     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detect faces in the grayscale image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # Check if faces were detected
# if len(faces) == 0:
#     print("No faces detected.")
# else:
#     print(f"Number of faces detected: {len(faces)}")

# faces=[]
# # Process each detected face
# for (index, (x, y, w, h)) in enumerate(faces):
#     # Draw rectangle around the face
#     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     # Print properties of the detected face
#     print(f"Face {index+1}:")
#     print(f"  - Coordinates (x, y): ({x}, {y})")
#     print(f"  - Dimensions (width, height): ({w}, {h})")
#     print(f"  - Aspect Ratio: {w / h:.2f}")

#     faces.append((x,y,w,h))

# print(faces)

#  # Crop the face from the image
# cropped_face = image[y:y+h, x:x+w]
# # Save the cropped face
# face_filename = f"face_{index+1}.jpg"  # Save each face with a unique filename
# cv2.imwrite(face_filename, cropped_face)
# print(f"Cropped face {index+1} saved as {face_filename}")

# # Display the image with rectangles around the detected faces
# cv2.imshow('Image with Detected Faces', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import os

# Paths to cascade and image
cascade_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\haarcascade-frontalface-default.xml'
image_path = 'D:\\Lenditt\\Face_Recognoition\\Detect_face\\1.jpg'

# Check if files exist
if not os.path.isfile(cascade_path):
    print(f"Error: Haar cascade file '{cascade_path}' not found.")
    exit(1)

if not os.path.isfile(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    exit(1)

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier(cascade_path)

# Read the input image
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is None:
    print(f"Error: Unable to read image file '{image_path}'.")
    exit(1)

# Check image dimensions and rotate if necessary
h, w, _ = image.shape
if h > w:
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Check if faces were detected
if len(faces) == 0:
    print("No faces detected.")
else:
    print(f"Number of faces detected: {len(faces)}")

# Find the face with the largest area (width * height)
max_area = 0
max_face = None

for (index, (x, y, w, h)) in enumerate(faces):
    # Calculate area of the face
    area = w * h
    
    # Update max_face if current face has larger area
    if area > max_area:
        max_area = area
        max_face = (x, y, w, h)

# Check if a face with maximum area was found
if max_face is None:
    print("No faces found with maximum area.")
    exit(1)

# Extract coordinates and dimensions of the face with maximum area
x, y, w, h = max_face

# Draw rectangle around the face with maximum area
cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Crop the face from the image
cropped_face = image[y:y+h, x:x+w]

# Save the cropped face with the maximum area
face_filename = f"face.jpg"
cv2.imwrite(face_filename, cropped_face)
print(f"Largest face saved as {face_filename}")

# Display the image with rectangles around the detected faces
cv2.imshow('Image with Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
