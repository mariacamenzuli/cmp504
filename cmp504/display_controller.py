import cv2
import numpy as np
from mss import mss
from PIL import Image


class DisplayController:
    def __init__(self, static_templates):
        # imread 2nd param 0 indicates read image in grayscale
        # grayscale apparently simplifies search in OpenCV
        self.templates = {
            k: cv2.imread(v, 0) for (k, v) in static_templates.items()
        }

        self.monitor = {
            'top': 0, 'left': 0, 'width': 1920, 'height': 1080
        }
        self.screen = mss()

        self.frame = None

    def refresh_frame(self):
        self.frame = self.capture_frame()

    def capture_frame(self):
        screengrab_img = self.screen.grab(self.monitor)
        img = Image.frombytes('RGB', screengrab_img.size, screengrab_img.rgb)
        img = np.array(img)
        # while the MSS library will take screenshots using RGB colors, OpenCV uses BGR colors
        img = self.convert_rgb_to_bgr(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def read_image(self, path):
        return cv2.imread(path, 0)

    def convert_rgb_to_bgr(self, img):
        return img[:, :, ::-1]

    def match_template(self, img_grayscale, template, threshold=0.9):
        """
        Matches template image in a target grayscaled image
        """

        res = cv2.matchTemplate(img_grayscale, template, cv2.TM_CCOEFF_NORMED)
        matches = np.where(res >= threshold)
        return matches

    def find_template(self, name, image=None, threshold=0.9):
        if image is None:
            if self.frame is None:
                self.refresh_frame()

            image = self.frame

        return self.match_template(
            image,
            self.templates[name],
            threshold
        )
