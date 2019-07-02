import logging
from dataclasses import dataclass
from enum import Enum

import cmp504.image_processing as image_processing
import cv2
import numpy as np
import pytesseract
from PIL import Image
from matplotlib import pyplot
from mss import mss


class TemplateMatchingMethod(Enum):
    SQUARE_DIFFERENCE = cv2.TM_SQDIFF
    SQUARE_DIFFERENCE_NORMALIZED = cv2.TM_SQDIFF_NORMED
    CROSS_CORRELATION = cv2.TM_CCORR
    CROSS_CORRELATION_NORMALIZED = cv2.TM_CCORR_NORMED
    CORRELATION_COEFFICIENT = cv2.TM_CCOEFF
    CORRELATION_COEFFICIENT_NORMALIZED = cv2.TM_CCOEFF_NORMED


@dataclass
class TemplateMatch:
    top_left: (int, int)
    bottom_right: (int, int)

    def __post_init__(self):
        self.field_b = self.mid_point = self.__calculate_midpoint(self.top_left, self.bottom_right)

    @staticmethod
    def __calculate_midpoint(point1, point2):
        return int(round((point1[0] + point2[0]) / 2)), int(round((point1[1] + point2[1]) / 2))


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
        self.frame = self.__convert_rgb_to_bgr(img)

    def load_frame(self, frame_path: str):
        logging.debug("Loading frame from file '%s'.", frame_path)
        self.frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)

    def crop_frame(self, top_left, bottom_right):
        self.frame = self.frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()

    def find_text(self, image=None, pre_processing_chain: image_processing.ImageProcessingStepChain = None):
        if image is None:
            target_image = self.frame
        else:
            target_image = image

        if pre_processing_chain is not None:
            target_image = pre_processing_chain.apply(target_image)

        return pytesseract.image_to_string(target_image)

    def find_template_matches(self,
                              template_path: str,
                              threshold: float = 0.9,
                              method: TemplateMatchingMethod = TemplateMatchingMethod.CORRELATION_COEFFICIENT_NORMALIZED,
                              render_matches: bool = False):
        logging.debug("Looking for template match using template from file '%s', threshold %f, and matching method %s.",
                      template_path,
                      threshold,
                      method.name)
        self.assert_controller_has_frame()

        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        template_height = template.shape[0]
        template_width = template.shape[1]

        template_split = self.split_out_alpha_mask(template)
        if template_split['mask_present'] is True:
            match_result = cv2.matchTemplate(self.frame, template_split['image'], method.value, template_split['mask'])
        else:
            match_result = cv2.matchTemplate(self.frame, template_split['image'], method.value)

        # The best match for SQUARE_DIFFERENCE and SQUARE_DIFFERENCE_NORMALIZED is the global minimum value.
        # The best match for the other methods is the global maximum value.
        if method in [TemplateMatchingMethod.SQUARE_DIFFERENCE,
                      TemplateMatchingMethod.SQUARE_DIFFERENCE_NORMALIZED]:
            match_locations = np.where(match_result <= threshold)
        else:
            match_locations = np.where(match_result >= threshold)

        frame_copy = self.frame.copy()
        matches = []
        for match_location in zip(*match_locations[::-1]):
            matches.append(TemplateMatch(match_location,
                                         (match_location[0] + template_width, match_location[1] + template_height)))
            cv2.rectangle(frame_copy,
                          match_location,
                          (match_location[0] + template_width, match_location[1] + template_height),
                          255,
                          2)

        if render_matches:
            self.render_image(frame_copy)

        return matches

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

        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        template_height = template.shape[0]
        template_width = template.shape[1]

        template_split = self.split_out_alpha_mask(template)
        if template_split['mask_present'] is True:
            match_result = cv2.matchTemplate(self.frame, template_split['image'], method.value, template_split['mask'])
        else:
            match_result = cv2.matchTemplate(self.frame, template_split['image'], method.value)

        min_value, max_value, min_location, max_location = cv2.minMaxLoc(match_result)

        # The best match for SQUARE_DIFFERENCE and SQUARE_DIFFERENCE_NORMALIZED is the global minimum value.
        # The best match for the other methods is the global maximum value.
        if method in [TemplateMatchingMethod.SQUARE_DIFFERENCE, TemplateMatchingMethod.SQUARE_DIFFERENCE_NORMALIZED]:
            logging.info("Minimum match value was %f.", min_value)
            if min_value >= threshold:
                logging.info("No match found.")
                return None

            top_left = min_location
        else:
            logging.info("Maximum match value was %f.", max_value)
            if max_value <= threshold:
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

        return TemplateMatch(top_left, bottom_right)

    @staticmethod
    def split_out_alpha_mask(image):
        if image.shape[2] > 3:
            logging.debug('Alpha channel detected in image. Splitting out mask.')
            channels = cv2.split(image)
            mask = np.array(channels[3])
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            return {'mask_present': True, 'image': image, 'mask': mask}
        else:
            return {'mask_present': False, 'image': image}

    def render_frame(self):
        self.render_image(self.frame)

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
