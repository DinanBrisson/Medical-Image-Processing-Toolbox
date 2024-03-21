import cv2 as cv

class SmileDetector:
    def detect_smile(self):
        # Load classifier
        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')
        eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

        # Capturing video from webcam
        cap = cv.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Grayscale conversion
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Faces detecion
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv.putText(frame, 'Face', (x + w, y + h + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # ROI for smile and eye detection
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
                for (sx, sy, sw, sh) in smiles:
                    # Rectangle around detected smile
                    cv.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                    cv.putText(frame, 'Smile', (x + sx, y + sy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
                for (ex, ey, ew, eh) in eyes:
                    # Rectangle around detected eyes
                    cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
                    cv.putText(frame, 'Eye', (x + ex, y + ey + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv.imshow('Smile detector', frame)

            # Exit loop if 'q' key is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Liberation of resources
        cap.release()
        cv.destroyAllWindows()