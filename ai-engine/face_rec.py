import cv2
import face_recognition

def detect_vip_customer(frame):
    # Load known faces
    known_image = face_recognition.load_image_file('vip_database/elon_musk.jpg')
    biden_encoding = face_recognition.face_encodings(known_image)[0]
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces([biden_encoding], face_encoding)
        if match[0]:
            return 'VIP DETECTED: ELON MUSK'
    return 'Unknown'

