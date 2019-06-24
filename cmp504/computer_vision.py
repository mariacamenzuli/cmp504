import logging
import cv2
import numpy as np
from mss import mss
from PIL import Image
from matplotlib import pyplot
from enum import Enum


class TemplateMatchingMethod(Enum):
    SQUARE_DIFFERENCE = cv2.TM_SQDIFF
    SQUARE_DIFFERENCE_NORMALIZED = cv2.TM_SQDIFF_NORMED
    CROSS_CORRELATION = cv2.TM_CCORR
    CROSS_CORRELATION_NORMALIZED = cv2.TM_CCORR_NORMED
    CORRELATION_COEFFICIENT = cv2.TM_CCOEFF
    CORRELATION_COEFFICIENT_NORMALIZED = cv2.TM_CCOEFF_NORMED


class CVController:
    def __init__(self):
        self.screen = mss()
        self.frame = None

    def capture_frame(self, monitor_number: int = 1):
        logging.debug("Capturing frame from monitor %d.", monitor_number)
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

    def load_frame(self, frame_path: str):
        logging.debug("Loading frame from file '%s'.", frame_path)
        self.frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

    # todo: add method to find all matches
    def find_template_match(self,
                            template_path: str,
                            threshold: float = 0.9,
                            method: TemplateMatchingMethod = TemplateMatchingMethod.CORRELATION_COEFFICIENT_NORMALIZED,
                            render_match: bool = False):
        logging.debug("Looking for template match using template from file '%s', threshold %f, and matching method %s.",
                      template_path,
                      threshold,
                      method.name)
        self.assert_controller_has_frame()

        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        template_width, template_height = template.shape[::-1]

        match_result = cv2.matchTemplate(self.frame, template, method.value)
        min_value, max_value, min_location, max_location = cv2.minMaxLoc(match_result)

        # The best match for SQUARE_DIFFERENCE and SQUARE_DIFFERENCE_NORMALIZED is the global minimum value.
        # The best match for the other methods is the global maximum value.
        if method in [TemplateMatchingMethod.SQUARE_DIFFERENCE, TemplateMatchingMethod.SQUARE_DIFFERENCE_NORMALIZED]:
            logging.info("Minimum match value was %f.", min_value)
            if min_value > threshold:
                logging.info("No match found.")
                return None

            top_left = min_location
        else:
            logging.info("Maximum match value was %f.", max_value)
            if max_value < threshold:
                logging.info("No match found.")
                return None

            top_left = max_location

        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

        logging.info("Match found in patch with top left corner at (%d, %d), and bottom right corner at (%d, %d).",
                     top_left[0],
                     top_left[1],
                     bottom_right[0],
                     bottom_right[1])

        if render_match:
            frame_copy = self.frame.copy()
            cv2.rectangle(frame_copy, top_left, bottom_right, 255, 2)
            self.render_image(frame_copy)

        return self.__calculate_midpoint(top_left, bottom_right)

    @staticmethod
    def render_image(image):
        pyplot.imshow(image, cmap='gray')
        pyplot.title('Template Match'), pyplot.xticks([]), pyplot.yticks([])
        pyplot.show()

    def assert_controller_has_frame(self):
        assert (self.frame is not None), "A frame is required. Use 'capture_frame' or 'load_frame' to prepare frame."

    @staticmethod
    def __convert_rgb_to_bgr(image):
        return image[:, :, ::-1]

    @staticmethod
    def __calculate_midpoint(point1, point2):
        return int(round((point1[0] + point2[0]) / 2)), int(round((point1[1] + point2[1]) / 2))
