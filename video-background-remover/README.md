# Video Background Remover

## Overview
The Video Background Remover project aims to provide a solution for real-time mobile recording using GoPro or DJI Action cameras. The application removes moving objects, such as people and animals, from the video stream, allowing users to capture a background-only video.

## Features
- Real-time video capture from DJI Action 5 Pro camera.
- Automatic removal of moving objects using motion estimation and semantic segmentation.
- Background reconstruction to fill gaps left by removed objects.
- User-friendly GUI for easy interaction.

## Project Structure
```
video-background-remover
├── src
│   ├── main.py                # Entry point of the application
│   ├── gui
│   │   └── app_window.py      # Main GUI definition
│   ├── processing
│   │   ├── motion_estimation.py # Frame preprocessing and motion detection
│   │   ├── segmentation.py      # Semantic segmentation functions
│   │   ├── fusion_engine.py     # Combines motion and segmentation results
│   │   └── background_inpainting.py # Background reconstruction functions
│   └── capture
│       └── camera_stream.py     # Handles camera connection and video frames
├── requirements.txt            # Project dependencies
├── .gitignore                  # Files to ignore in version control
└── README.md                   # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/video-background-remover.git
   ```
2. Navigate to the project directory:
   ```
   cd video-background-remover
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```
   python src/main.py
   ```
2. Use the GUI to start and stop video capture.
3. The processed video will display in the application window with moving objects removed.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.