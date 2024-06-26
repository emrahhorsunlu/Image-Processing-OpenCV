import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QSizePolicy, QLineEdit, QColorDialog
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from PyQt5.QtCore import Qt

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super(ImageProcessingApp, self).__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.original_image_label = QLabel(self)
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.original_image_label.setScaledContents(True)

        self.processed_image_label = QLabel(self)
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.processed_image_label.setScaledContents(True)

        self.original_image_text_label = QLabel('Orijinal Resim', self)
        self.original_image_text_label.setAlignment(Qt.AlignCenter)
        self.original_image_text_label.setFont(QFont('Arial', 14, QFont.Bold))

        self.processed_image_text_label = QLabel('Dönüştürülmüş Resim', self)
        self.processed_image_text_label.setAlignment(Qt.AlignCenter)
        self.processed_image_text_label.setFont(QFont('Arial', 14, QFont.Bold))

        self.load_image_button = QPushButton('Resim Yükle', self)
        self.load_image_button.clicked.connect(self.load_image)
        self.load_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.load_image_button.setStyleSheet("background-color: #3498db; color: white; font-size: 14px; font-weight: bold;")

        self.process_combo_box = QComboBox(self)
        self.process_combo_box.addItems(['Border', 'Blur', 'Sharpen', 'Gamma', 'Sobel', 'Harris Corner', 'Face Detection'])
        self.process_combo_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.process_combo_box.setStyleSheet("background-color: #2ecc71; color: white; font-size: 14px; font-weight: bold;")

        self.blur_label = QLabel('Blur Değeri:', self)
        self.blur_input = QLineEdit(self)
        self.blur_input.setPlaceholderText("Örn: 5")
        self.blur_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.blur_input.setStyleSheet("font-size: 12px;")

        self.border_thickness_label = QLabel('Border Kalınlığı:', self)
        self.border_thickness_input = QLineEdit(self)
        self.border_thickness_input.setPlaceholderText("Örn: 10")
        self.border_thickness_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.border_thickness_input.setStyleSheet("font-size: 12px;")

        self.border_color_label = QLabel('Border Renk Kodu:', self)
        self.border_color_button = QPushButton('Renk Seç', self)
        self.border_color_button.clicked.connect(self.get_border_color)
        self.border_color_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.border_color_button.setStyleSheet("background-color: #3498db; color: white; font-size: 12px;")

        self.process_image_button = QPushButton('İşlem Yap', self)
        self.process_image_button.clicked.connect(self.process_image)
        self.process_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.process_image_button.setStyleSheet("background-color: #e74c3c; color: white; font-size: 14px; font-weight: bold;")

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.central_widget.setStyleSheet("background-color: #ecf0f1;")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_image_button)
        button_layout.addWidget(self.process_combo_box)
        button_layout.addWidget(self.blur_label)
        button_layout.addWidget(self.blur_input)
        button_layout.addWidget(self.border_thickness_label)
        button_layout.addWidget(self.border_thickness_input)
        button_layout.addWidget(self.border_color_label)
        button_layout.addWidget(self.border_color_button)
        button_layout.addWidget(self.process_image_button)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.processed_image_label)

        text_label_layout = QHBoxLayout()
        text_label_layout.addWidget(self.original_image_text_label)
        text_label_layout.addWidget(self.processed_image_text_label)

        self.layout.addLayout(text_label_layout)
        self.layout.addLayout(image_layout)
        self.layout.addLayout(button_layout)

        self.original_image = None
        self.processed_image = None

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        

    def load_image(self):
        file_dialog = QFileDialog()
        file_name, _ = file_dialog.getOpenFileName(self, "Resim Seç", "", "Image Files (*.png *.jpg *.bmp *.jpeg);;All Files (*)")

        if file_name:
            self.original_image = cv2.imread(file_name)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_original_image()

    def process_image(self):
        if self.original_image is not None:
            selected_index = self.process_combo_box.currentIndex()

            blur_value = self.blur_input.text()
            border_thickness = self.border_thickness_input.text()

            try:
                blur_value = float(blur_value)
                border_thickness = int(border_thickness)
            except ValueError:
                blur_value = None
                border_thickness = None

            

            self.processed_image = self.original_image.copy()

            if selected_index == 0:  # Border
                border_color = self.get_border_color()
                if border_thickness is not None and border_color is not None:
                    self.processed_image = cv2.copyMakeBorder(self.processed_image, border_thickness, border_thickness, border_thickness, border_thickness,
                                                              borderType=cv2.BORDER_CONSTANT, value=border_color)
            elif selected_index == 1:  # Blur
                if blur_value is not None:
                    kernel_size = int(blur_value)
                    self.processed_image = cv2.blur(self.processed_image, (kernel_size, kernel_size))
            elif selected_index == 2:  # Sharpen
                self.processed_image = cv2.filter2D(self.processed_image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
            elif selected_index == 3:  # Gamma
                gamma = 1.5
                self.processed_image = np.power(self.processed_image / float(np.max(self.processed_image)), gamma) * 255.0
            elif selected_index == 4:  # Sobel
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                magnitude = np.uint8(magnitude)
                self.processed_image = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
            elif selected_index == 5:  # Harris Corner
                gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                gray_image = np.float32(gray_image)
                dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)
                dst = cv2.dilate(dst, None)
                self.processed_image[dst > 0.01 * dst.max()] = [0, 0, 255]
            elif selected_index == 6:  # Face Detection
                self.detect_faces()

            self.display_processed_image()

    def display_original_image(self):
        if self.original_image is not None:
            height, width, channel = self.original_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.original_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image)
            label_width = 400
            label_height = int(height * (label_width / width))
            self.original_image_label.setPixmap(pixmap)
            self.original_image_label.setFixedSize(label_width, label_height)

    def display_processed_image(self):
        if self.processed_image is not None:
            height, width, channel = self.processed_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.processed_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image)
            label_width = 400
            label_height = int(height * (label_width / width))
            self.processed_image_label.setPixmap(pixmap)
            self.processed_image_label.setFixedSize(label_width, label_height)

    def detect_faces(self):
        gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(self.processed_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    def get_border_color(self):
        color_dialog = QColorDialog(self)
        color = color_dialog.getColor()
        if color.isValid():
            return color.getRgb()[:3]
        else:
            return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
