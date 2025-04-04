import cv2

ext_cam = False

url = 'http://192.168.11.135:8080/video'

# Étape 1 : Capture vidéo
def capture_video():
    cap = cap = cv2.VideoCapture(url) if ext_cam else cv2.VideoCapture(0)
    return cap

# Étape 2 : Détection de visages
def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Étape 3 : Affichage des résultats
def display_results(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Armel", (x, y-10), 2, 1, (0, 255, 0), 1)
        
    cv2.imshow("Face Detection", frame)

# Pipeline principal
def main():
    cap = capture_video()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)  # Appliquer la détection de visages
        display_results(frame, faces)  # Afficher les résultats

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()