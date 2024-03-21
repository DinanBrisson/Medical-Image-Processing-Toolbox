import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

    def convert_to_gray(self):
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray Image', gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_rgb_channels(self):
        # Extract the individual color channels
        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]

        # Display each color channel separately using matplotlib
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(red, cmap="Reds")
        ax.set_title("Red")
        ax.axis("off")
        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(green, cmap="Greens")
        ax.set_title("Green")
        ax.axis("off")
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(blue, cmap="Blues")
        ax.set_title("Blue")
        ax.axis("off")
        plt.show()

    def invert_image(self):
        # Invert the colors of the image
        inverted_image = cv2.bitwise_not(self.image)
        # Display the inverted image
        cv2.imshow('Inverted Image', inverted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_to_gray_and_invert(self):
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Invert the grayscale image
        inverted_gray_image = cv2.bitwise_not(gray_image)
        # Display the inverted grayscale image
        cv2.imshow('Inverted Gray Image', inverted_gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def select_and_apply_roi(self):
        # Select a region of interest (ROI) from the image
        roi = cv2.selectROI(self.image)
        x, y, w, h = roi
        roi_extracted = self.image[y:y + h, x:x + w]

        # Apply Laplace edge detection to the ROI
        kernel_laplace = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
        roi_laplace = cv2.filter2D(roi_extracted, -1, kernel_laplace)

        # Replace the ROI with the Laplacian-filtered ROI in the original image
        image_with_roi_replaced = self.image.copy()
        image_with_roi_replaced[y:y + h, x:x + w] = roi_laplace

        # Display the image with the replaced ROI
        cv2.imshow('Image with ROI Replaced', image_with_roi_replaced)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def segment_lesion(self):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding to segment the lesion
        block_size = 15
        constant_c = 5
        thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                  block_size, constant_c)
        # Display the segmented lesion
        cv2.imshow('Segmented Lesion', thresholded_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
