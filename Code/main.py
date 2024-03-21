import sys
from PyQt5.QtWidgets import QApplication
from window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 400, 600)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()