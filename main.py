import cv2
import os

def add_new_student():
    print("---Add new student name----")
    student_name=input("Enter Student name: -").strip()

    if not os.path.exists('faces_dataset'):
       os.makedirs('faces_dataset')
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: could not open webcam")
        return
    print("Capturing img in 3 sec.......")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frames")
            break

        # preprossing : convert img to black and white
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"faces_dataset/{student_name}_{count}.jpg", face)
            count += 1
        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
           break

    cam.release()
    cv2.destroyAllWindows()


add_new_student()


       
