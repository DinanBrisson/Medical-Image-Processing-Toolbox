import cv2
import numpy as np

class VesselDetector:
    def __init__(self, video_path):
        self.video_path = video_path

    def preprocess_image(self, image):
        # Normalize image
        normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Reduce noise - Gaussian blur filter
        blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)
        return blurred_image

    def detect_vessel_contours(self, image):
        # Convert to rayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect vessel contours using the Canny algorithm
        edges = cv2.Canny(gray_image, 30, 150)
        return edges

    def binarize_image(self, image):
        # Binarize the image using the Otsu method
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image

    def clean_binary_mask(self, binary_mask):
        # Apply a closing operation to remove noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        return cleaned_mask

    def detect_vessels(self):
        video = cv2.VideoCapture(self.video_path)

        while True:
            ret, frame = video.read()

            if not ret:
                break

            # Preprocess the frame
            preprocessed_frame = self.preprocess_image(frame)
            # Detect vessel contours
            vessel_contours = self.detect_vessel_contours(preprocessed_frame)
            # Binarize the vessel contours image
            binary_image = self.binarize_image(vessel_contours)
            # Clean the binary mask
            cleaned_mask = self.clean_binary_mask(binary_image)

            # Copy the original frame to highlight the detected vessels
            highlighted_vessels = frame.copy()
            highlighted_vessels[cleaned_mask == 255] = [0, 255, 0]

            # Display the frame with the highlighted vessels
            cv2.imshow('Fluorescent Vessels', highlighted_vessels)

            # Check for the 'q' key to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video resource
        video.release()
        cv2.destroyAllWindows()
