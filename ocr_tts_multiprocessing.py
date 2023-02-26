import cv2
import numpy as np
import pytesseract
import pyttsx3
from deskew import determine_skew

import time
import requests
from multiprocessing import Process, shared_memory, Value
from configparser import ConfigParser

# Converts RGB image to grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Corrects the nonuniform illumination of the background (useful before binarization)
def background_correction(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    divide = cv2.divide(gray, bg, scale=255)
    return divide

# Uses Otsu thresholding to binarize a grayscale image
def thresholding(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Makes the text thinner
def erosion(gray):
    kernel = np.ones((1, 1),np.uint8)
    gray = cv2.bitwise_not(gray)
    gray = cv2.erode(gray, kernel, iterations=1)
    gray = cv2.bitwise_not(gray)
    return gray

# Deskews an image without resizing. Background color can be from 0 to 255
def undo_skew(image, background_color):
    angle = determine_skew(image, angle_pm_90=True)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], borderValue=background_color)

# Image preprocessing pipeline
def preprocessing(img):
    img = grayscale(img)
    img = background_correction(img)
    img = thresholding(img)
    img = erosion(img)
    img = undo_skew(img, 255)
    return img

class CameraClass():
    def __init__(self, camera_id, seconds_between_ocr=3, display_regular_video=False, display_processed_video=True, perform_tts=True):
        self.camera_id = camera_id
        self.seconds_between_ocr = seconds_between_ocr
        self.display_regular_video = display_regular_video
        self.display_processed_video = display_processed_video
        self.perform_tts = perform_tts

        # Temporary camera object to get height and width of the video
        temp_camera = cv2.VideoCapture(camera_id)
        if temp_camera.isOpened():
            self.height = int(temp_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(temp_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            temp_camera.release()
        else:
            print("Error opening camera. Exiting program.")
            temp_camera.release()
            exit(1)

        # 1: All processes will continue running
        # 0: All processes will stop running
        self.run = Value('i', 1)

        # Allocating shared memory where latest frame will be stored
        self.shm_frame = shared_memory.SharedMemory(create=True, size=self.height*self.width*3)
        # Allocating shared memory where latest processed frame will be stored
        self.shm_processed = shared_memory.SharedMemory(create=True, size=self.height*self.width)

    # Process to continuously read frames from the camera stream
    def read(self):
        frame = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.shm_frame.buf)
        camera = cv2.VideoCapture(self.camera_id)
        create_process = True

        while bool(self.run.value):
            if camera.isOpened():
                ret, frame_read = camera.read()
                if not ret:
                    self.run.value = 0
                    break
                
                frame[:] = frame_read[:]
                if create_process:
                    Process(target=self.display).start()
                    create_process = False
            else:
                print("Error opening camera. Exiting program.")
                self.run.value = 0

        camera.release()

    # Process to preprocess latest frame from the camera stream, and display that processed frame
    def display(self):
        frame = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.shm_frame.buf)
        processed = np.ndarray((self.height, self.width), dtype=np.uint8, buffer=self.shm_processed.buf)
        create_process = True

        while bool(self.run.value):
            processed[:] = preprocessing(frame)
            if create_process:
                Process(target=self.ocr).start()
                create_process = False

            if self.display_regular_video:
                cv2.imshow('Regular', frame)

            if self.display_processed_video:
                cv2.imshow('Processed', processed)
            
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                self.run.value = 0

    # Process to perform OCR every N seconds on latest processed frame
    def ocr(self):
        processed = np.ndarray((self.height, self.width), dtype=np.uint8, buffer=self.shm_processed.buf)

        while bool(self.run.value):
            time.sleep(self.seconds_between_ocr)
            text = pytesseract.image_to_string(processed, lang='eng').strip()
            if text:
                print(text)
                if self.perform_tts:
                    proc = Process(target=self.tts, args=(text,))
                    proc.start()
                    while proc.is_alive():
                        if not bool(self.run.value):
                            proc.terminate()
                            proc.join()

    # Process to perform Text to Speech on given text
    def tts(self, text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    def start(self):
        proc = Process(target=self.read)
        proc.start()
        proc.join()
        self.shm_frame.unlink()
        self.shm_processed.unlink()
                
if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')

    ip_port = config['Settings']['ip_port']
    url = f'http://{ip_port}/video'

    webcam_id = int(config['Settings']['webcam_id'])

    try:
        print("Trying to connect to IP Webcam server...")
        response = requests.head(url, timeout=3)
        if response.status_code == 200:
            print("Connection established.")
            id = url
        else:
            print('Error connecting to server. Make sure the server is online, and the server address is correct. Defaulting to web cam.')
            id = webcam_id
    except:
        print('Error connecting to server. Make sure the server is online, and the server address is correct. Defaulting to web cam.')
        id = webcam_id

    # camera = CameraClass(camera_id=id, seconds_between_ocr=3, display_regular_video=True, display_processed_video=True, perform_tts=True)
    camera = CameraClass(camera_id=id)
    camera.start()
