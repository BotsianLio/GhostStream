import sys
from PyQt5.QtWidgets import QApplication, QDialog
from gui.app_window import AppWindow
from gui.selector import CameraSelector

def main():
    app = QApplication(sys.argv)
    
    # 1. Launch Selector First
    selector = CameraSelector()
    
    if selector.exec_() == QDialog.Accepted:
        selected_index = selector.selected_index
        
        # 2. Pass index to AppWindow (It will handle the Thread/Worker)
        window = AppWindow(selected_index)
        window.show()
        
        sys.exit(app.exec_())
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()