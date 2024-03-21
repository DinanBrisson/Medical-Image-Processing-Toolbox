import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage import exposure

class DicomProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_display_dicom_image(self):
        # Load the DICOM image
        dicom_image = pydicom.dcmread(self.file_path)

        # Display image information
        print(dicom_image)

        # Check if the image has multiple axial slices
        if hasattr(dicom_image, "NumberOfFrames"):
            num_frames = dicom_image.NumberOfFrames
            print("Number of axial slices:", num_frames)
        else:
            print("The DICOM image does not contain information about the number of axial slices.")

        # Retrieve pixel values
        pixel_array = dicom_image.pixel_array

        # Min-Max normalization using scikit-image
        normalized_image = exposure.rescale_intensity(pixel_array, in_range='image', out_range=(0, 1))

        # Contrast adjustment
        contrast_adjusted_image = exposure.equalize_adapthist(normalized_image)

        # Display both images side by side
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(dicom_image.pixel_array, cmap='gray')
        plt.title("Original DICOM Image")

        # Normalized image
        plt.subplot(1, 2, 2)
        plt.imshow(contrast_adjusted_image, cmap='gray')
        plt.title("Normalized DICOM Image")

        # Show the images
        plt.show()
