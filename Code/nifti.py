import cv2
import numpy as np
from skimage.morphology import disk, dilation
from scipy import ndimage
import nibabel as nib

class NiftiProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def display_views(self):
        # Load the NIfTI medical image
        medical_image = nib.load(self.file_path)
        image = medical_image.get_fdata()

        # Sagittal Plane
        sagittal_image = np.flip(image[int(image.shape[0] // 2), :, :], axis=0)
        sagittal_view = np.rot90(sagittal_image)
        cv2.imshow("Sagittal Plane", sagittal_view)

        # Axial (Transversal) Plane
        axial_image = np.rot90(image[:, :, int(image.shape[2] // 2)], k=2)
        cv2.imshow("Axial Plane", axial_image)

        # Coronal (Frontal) Plane
        coronal_image = np.rot90(image[:, int(image.shape[1] // 2), :], k=1)
        cv2.imshow("Coronal Plane", coronal_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_slice(self, index):
        # Load the NIfTI medical image
        medical_image = nib.load(self.file_path)
        image = medical_image.get_fdata()

        # Valid index
        if index < 0 or index >= image.shape[2]:
            raise ValueError("Invalid slice index")

        # Retrieve the axial slice corresponding to the chosen index
        axial_slice = np.rot90(image[:, :, index], k=2)

        return axial_slice

    def choose_axial_slice(self, index=None):
        # Load the NIfTI medical image
        medical_image = nib.load(self.file_path)
        image = medical_image.get_fdata()

        if index is None:
            index = image.shape[2] // 2
        else:
            if index < 0 or index >= image.shape[2]:
                raise ValueError("Invalid slice index")

        return index

    def standardization(self, image, brain_mask):
        mean = np.mean(image)
        std = np.std(image)
        normalized_image = (image - mean) / std
        return normalized_image

    def create_brain_mask(self, image, threshold_value):
        # Background mask using a threshold
        background_mask = (image > threshold_value).astype(np.uint8)

        # Dilatation
        disk_radius = 3
        selem = disk(disk_radius)
        dilated_background_mask = dilation(background_mask, selem)

        # Label the dilated mask
        labeled_mask, num_labels = ndimage.label(dilated_background_mask)

        # Identify the brain-containing regions
        # Assuming the largest region (after background) corresponds to the brain
        unique_labels, label_counts = np.unique(labeled_mask, return_counts=True)
        label_counts = label_counts[1:]  # Ignore background (label 0)
        largest_label = unique_labels[np.argmax(label_counts)]

        # Create the brain mask based on the most frequent (largest) label
        brain_mask = np.where(labeled_mask == largest_label, 1, 0).astype(np.uint8)

        return brain_mask

    def display_normalized_nifti_image(self, threshold_value=50):
        index = self.choose_axial_slice()
        slice_image = self.get_slice(index)
        brain_mask = self.create_brain_mask(slice_image, threshold_value)
        normalized_image = self.standardization(slice_image, brain_mask)

        cv2.imshow("Normalized NIfTI Image", normalized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
