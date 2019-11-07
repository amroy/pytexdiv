from PyQt5.QtWidgets import QMainWindow, QApplication
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Texture Retrieval Model Performance Analyzer"
        self.width = 700
        self.height = 400

        self.init_window()
        self.show()

    def init_window(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, self.width, self.height)


app = QApplication(sys.argv)
window = MainWindow()
sys.exit(app.exec())
