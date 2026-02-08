import sys
from PyQt5.QtWidgets import QApplication, QDialog
from gui.app_window import AppWindow
from gui.selector import CameraSelector 
from capture.camera_stream import CameraStream

def main():
    app = QApplication(sys.argv)
    
    # 1. Launch Simple Selector (No Preview)
    selector = CameraSelector()
    
    if selector.exec_() == QDialog.Accepted:
        selected_index = selector.selected_index
        print(f"User selected Camera Index: {selected_index}")
        
        # 2. Start Main App with chosen index
        camera_stream = CameraStream(source=selected_index)
        window = AppWindow(camera_stream)
        window.show()
        
        sys.exit(app.exec_())
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()