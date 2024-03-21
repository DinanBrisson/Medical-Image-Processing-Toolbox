import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt
from image import ImageProcessor
from dicom import DicomProcessor
from nifti import NiftiProcessor
from vessel import VesselDetector
from detector import SmileDetector


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interface")

        self.image_processor = ImageProcessor("../Dataset/jpg/Cellules.jpg")
        self.dicom_processor = DicomProcessor("../Dataset/DICOM/1.dcm")
        self.nifti_processor = NiftiProcessor("../Dataset/NIfTI/1.nii")
        self.vessel_detector = VesselDetector("../Dataset/Angiography/Angiography.avi")
        self.smile_detector = SmileDetector()
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Text
        label = QLabel("Click the buttons to perform the features.<br>"
                        "Exit an image with the q button.<br>"
                        "Select a ROI with the mouse then validate with enter.<br>"
                       )
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Buttons
        btn_gray = QPushButton("Convert to Gray")
        btn_gray.clicked.connect(self.image_processor.convert_to_gray)
        layout.addWidget(btn_gray)

        btn_rgb = QPushButton("Display RGB Channels")
        btn_rgb.clicked.connect(self.image_processor.display_rgb_channels)
        layout.addWidget(btn_rgb)

        btn_invert = QPushButton("Invert Image")
        btn_invert.clicked.connect(self.image_processor.invert_image)
        layout.addWidget(btn_invert)

        btn_roi = QPushButton("Select and Apply ROI")
        btn_roi.clicked.connect(self.image_processor.select_and_apply_roi)
        layout.addWidget(btn_roi)

        btn_segment = QPushButton("Segment Lesion")
        btn_segment.clicked.connect(self.image_processor.segment_lesion)
        layout.addWidget(btn_segment)

        btn_dicom = QPushButton("Load and Display DICOM Image")
        btn_dicom.clicked.connect(self.dicom_processor.load_and_display_dicom_image)
        layout.addWidget(btn_dicom)

        btn_nifti = QPushButton("Display NIfTI Views")
        btn_nifti.clicked.connect(self.nifti_processor.display_views)
        layout.addWidget(btn_nifti)

        btn_normalize = QPushButton("Normalize NIfTI Image")
        btn_normalize.clicked.connect(self.nifti_processor.display_normalized_nifti_image)
        layout.addWidget(btn_normalize)

        btn_vessel = QPushButton("Detect Vessels")
        btn_vessel.clicked.connect(self.vessel_detector.detect_vessels)
        layout.addWidget(btn_vessel)

        btn_smile = QPushButton("Detect Smiles")
        btn_smile.clicked.connect(self.smile_detector.detect_smile)
        layout.addWidget(btn_smile)

        self.show()
