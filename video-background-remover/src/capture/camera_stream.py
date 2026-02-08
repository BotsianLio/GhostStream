import cv2

class CameraStream:
    def __init__(self, source=0):
        """
        source: 
            int: Camera index for USB connection (e.g., 0, 1).
            str: URL for wireless stream (e.g., 'rtmp://192.168.1.5/live/stream').
        """
        self.source = source
        self.cap = None

    def start_stream(self):
        # cv2.VideoCapture handles both integers (USB) and strings (RTMP/RTSP/HTTP) automatically
        self.cap = cv2.VideoCapture(self.source)
        
        # Note: Resolution settings often don't apply to network streams (they are fixed by the sender),
        # but we can try setting buffer size to minimize latency.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.source}")

    def get_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def stop_stream(self):
        if self.cap is not None:
            self.cap.release()