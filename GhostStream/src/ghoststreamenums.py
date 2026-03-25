from enum import Enum

class FrameSourceType(Enum):
    VIDEO = 0
    CAMERA = 1

class MethodType(Enum):
    RANSAC = "Ransac"
    MAGSAC_PLUS_PLUS = "Magsac++"
    

