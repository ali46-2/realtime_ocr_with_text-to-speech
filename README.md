# Realtime OCR with TTS
A Python program that captures video from a camera, and performs realtime TTS (Text to Speech) on any text found in the video.

# Features
## Multiprocessing
This program uses multiprocessing to improve performance by allocating a separate process for each major task:
  * Capture video from the camera.
  * Preprocess the latest frame from the camera.
  * Perform OCR (Optical Character Regonition) every N seconds on the latest preprocessed frame.
  * Perform TTS if any text is found from the OCR process.
  
## Preprocessing
The preprocessing steps include:
  * Background color correction.
  * Binarizing the image using Otsu thresholding.
  * Deskewing the image to correct its orientation.

<div align="middle">
<img src="images/regular.png" width="360">
<img src="images/processed.png" width="360">
</div>
  
# Video Source
The program supports two sources for the video:
  * Streaming the video through a mobile phone's camera using the IP Webcam app.
  * Using a laptop's webcam or any physically attached webcam.
  
# Requirements
This program was tested on Python 3.9. All external packages used in this program are listed in the [requirements.txt](requirements.txt) file.

# Configuration
The server URL for the IP Webcam server, and the webcam id must be specified in the [config.ini](config.ini) file before running the program. If you do not intend to use IP Webcam, then leave the `ip_port` field to its default value. Do not remove the field though. The program will automatically try use the webcam if it notices that the server URL is incorrect or if the URL is not reachable.
