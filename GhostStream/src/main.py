import sys
from PyQt5.QtWidgets import QApplication, QDialog
from gui.app_window import AppWindow
from gui.selector import CameraSelector
from gui.video_window import VideoWindow

def main():
    app = QApplication(sys.argv)
    
    selector = CameraSelector()
    selector.show()
    sys.exit(app.exec_())
    if selector.exec_() == QDialog.Accepted:
        if selector.rval == "Camera": 
            selected_index = selector.selected_index

            window = AppWindow(selected_index)
            window.show()
        elif selector.rval == "Video": 
            video_path = selector.filepath

            window = VideoWindow(video_path)
            window.show()
        
        sys.exit(app.exec_())
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
