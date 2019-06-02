import cv2
import numpy as np
from mss import mss
from PIL import Image


class CVController:
    def __init__(self):
        self.screen = mss()
        self.frame = None

    def capture_frame(self, monitor_number=1):
        monitor_definition = self.screen.monitors[monitor_number]
        monitor_screenshot = self.screen.grab({
            "top": monitor_definition["top"],
            "left": monitor_definition["left"],
            "width": monitor_definition["width"],
            "height": monitor_definition["height"],
            "mon": monitor_number
        })

        img = Image.frombytes('RGB', monitor_screenshot.size, monitor_screenshot.rgb)
        img = np.array(img)
        # while the MSS library will take screenshots using RGB colors, OpenCV uses BGR colors
        img = self.__convert_rgb_to_bgr(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.frame = img_gray

    def load_frame(self, frame_path):
        self.frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

    def find_match(self, match_sought_path, threshold=0.9):
        match_sought = cv2.imread(match_sought_path, cv2.IMREAD_GRAYSCALE)
        res = cv2.matchTemplate(self.frame, match_sought, cv2.TM_CCOEFF_NORMED)
        matches = np.where(res >= threshold)
        return matches

    @staticmethod
    def __convert_rgb_to_bgr(image):
        return image[:, :, ::-1]
