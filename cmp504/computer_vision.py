import logging
from dataclasses import dataclass, field
from enum import Enum

import cmp504.image_processing as image_processing
import cv2
import sys
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


class HuTemplateMatchingMethod(Enum):
    METHOD_1 = cv2.CONTOURS_MATCH_I1
    METHOD_2 = cv2.CONTOURS_MATCH_I2  # Sum of absolute differences
    METHOD_3 = cv2.CONTOURS_MATCH_I3


@dataclass
class TemplateMatch:
    top_left: (int, int)
    bottom_right: (int, int)
    similarity_score: float = field(compare=False)
    threshold: float

    def __post_init__(self):
        self.mid_point = self.__calculate_midpoint(self.top_left, self.bottom_right)

    @staticmethod
    def __calculate_midpoint(point1, point2):
        return int(round((point1[0] + point2[0]) / 2)), int(round((point1[1] + point2[1]) / 2))


@dataclass
class FeatureBasedMatch:
    location: (int, int)
    distance_score: float
    threshold: float


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
                              method: TemplateMatchingMethod = TemplateMatchingMethod.CROSS_CORRELATION_NORMALIZED,
                              template_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                              frame_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                              render_matches: bool = False):
        logging.debug("Looking for template match using template from file '%s', threshold %f, and matching method %s.",
                      template_path,
                      threshold,
                      method.name)

        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        template_height = template.shape[0]
        template_width = template.shape[1]

        match_result = self.__match_template(template,
                                             method,
                                             template_pre_processing_chain,
                                             frame_pre_processing_chain)

        if self.is_best_match_the_global_minimum(method):
            match_locations = np.where(match_result <= threshold)
        else:
            match_locations = np.where(match_result >= threshold)

        frame_copy = self.frame.copy()
        matches = []
        for match_location in zip(*match_locations[::-1]):
            matches.append(TemplateMatch(match_location,
                                         (match_location[0] + template_width, match_location[1] + template_height),
                                         match_result[match_location[1]][match_location[0]],
                                         threshold))
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
                            method: TemplateMatchingMethod = TemplateMatchingMethod.CROSS_CORRELATION_NORMALIZED,
                            template_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                            frame_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                            match_horizontal_mirror: bool = False,
                            render_match: bool = False):
        logging.debug("Looking for template match using template from file '%s', threshold %f, and matching method %s.",
                      template_path,
                      threshold,
                      method.name)

        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        template_height = template.shape[0]
        template_width = template.shape[1]

        match_result_1 = self.__match_template(template,
                                               method,
                                               template_pre_processing_chain,
                                               frame_pre_processing_chain)

        # self.render_image(match_result_1)

        min_value, max_value, min_location, max_location = cv2.minMaxLoc(match_result_1)

        if match_horizontal_mirror:
            template_flipped_horizontally = image_processing.FlipHorizontal().process(template)

            match_result_2 = self.__match_template(template_flipped_horizontally,
                                                   method,
                                                   template_pre_processing_chain,
                                                   frame_pre_processing_chain)

            min_value2, max_value2, min_location2, max_location2 = cv2.minMaxLoc(match_result_2)

            if min_value2 < min_value:
                min_value = min_value2
                min_location = min_location2

            if max_value2 > max_value:
                max_value = max_value2
                max_location = max_location2

        if self.is_best_match_the_global_minimum(method):
            logging.info("Minimum match value was %f.", min_value)
            if min_value >= threshold:
                logging.info("No match found.")
                return None

            similarity_score = min_value
            top_left = min_location
        else:
            logging.info("Maximum match value was %f.", max_value)
            if max_value <= threshold:
                logging.info("No match found.")
                return None

            similarity_score = max_value
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
            self.render_image(frame_copy,
                              'Match top left (%d, %d), bottom right (%d, %d)' % (top_left[0],
                                                                                  top_left[1],
                                                                                  bottom_right[0],
                                                                                  bottom_right[1]))

        return TemplateMatch(top_left, bottom_right, similarity_score, threshold)

    def find_template_match_hu_moments(self,
                                       template_path: str,
                                       threshold: float = 0.5,
                                       method: HuTemplateMatchingMethod = HuTemplateMatchingMethod.METHOD_1,
                                       binarization_threshold: int = 127,
                                       stopping_threshold: float = 0,
                                       render_match: bool = False):
        self.__assert_controller_has_frame()

        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        template = image_processing.ColorTransparentPixels(image_processing.COLOR_WHITE).process(template)
        template = image_processing.BGR2Grayscale().process(template)
        template = image_processing.Threshold(binarization_threshold).process(template)
        template = image_processing.Invert().process(template)
        template_height = template.shape[0]
        template_width = template.shape[1]

        target_image = image_processing.BGR2Grayscale().process(self.frame)
        target_image = image_processing.Threshold(binarization_threshold).process(target_image)
        target_image = image_processing.Invert().process(target_image)

        # self.render_image(template, 'Template')
        # self.render_image(target_image, 'Target Image')

        target_image_height = target_image.shape[0]
        target_image_width = target_image.shape[1]

        if template_height > target_image_height or template_width > target_image_width:
            return None

        min_distance_found = sys.float_info.max
        min_top_left = (0, 0)
        min_bottom_right = (0, 0)
        match_found = False

        for x in range(0, target_image_width - template_width + 1):
            if min_distance_found <= stopping_threshold:
                break

            for y in range(0, target_image_height - template_height + 1):
                if min_distance_found <= stopping_threshold:
                    break

                top_left = (x, y)
                bottom_right = (x + template_width, y + template_height)

                distance = cv2.matchShapes(template,
                                           target_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]],
                                           method.value,
                                           1)

                if distance < min_distance_found:
                    match_found = True
                    min_distance_found = distance
                    min_top_left = top_left
                    min_bottom_right = bottom_right

        if render_match and match_found:
            frame_copy = self.frame.copy()
            cv2.rectangle(frame_copy, min_top_left, min_bottom_right, 255, 2)
            self.render_image(frame_copy,
                              'Match top left (%d, %d), bottom right (%d, %d), distance %03.2f' % (min_top_left[0],
                                                                                                   min_top_left[1],
                                                                                                   min_bottom_right[0],
                                                                                                   min_bottom_right[1],
                                                                                                   min_distance_found))

        if match_found and min_distance_found <= threshold:
            return TemplateMatch(min_top_left, min_bottom_right, min_distance_found, threshold)
        else:
            return None

    def find_template_match_hu_moments_custom(self,
                                              template_path: str,
                                              threshold: float = 0.5,
                                              binarization_threshold: int = 127,
                                              stopping_threshold: float = 0,
                                              render_match: bool = False):
        self.__assert_controller_has_frame()

        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        template = image_processing.ColorTransparentPixels(image_processing.COLOR_WHITE).process(template)
        template = image_processing.BGR2Grayscale().process(template)
        template = image_processing.Threshold(binarization_threshold).process(template)
        template = image_processing.Invert().process(template)
        template_height = template.shape[0]
        template_width = template.shape[1]

        target_image = image_processing.BGR2Grayscale().process(self.frame)
        target_image = image_processing.Threshold(binarization_threshold).process(target_image)
        target_image = image_processing.Invert().process(target_image)

        # self.render_image(template, 'Template')
        # self.render_image(target_image, 'Target Image')

        target_image_height = target_image.shape[0]
        target_image_width = target_image.shape[1]

        if template_height > target_image_height or template_width > target_image_width:
            return None

        min_distance_found = 999999999
        min_top_left = (0, 0)
        min_bottom_right = (0, 0)
        match_found = False

        template_hu_moments = cv2.HuMoments(cv2.moments(template))

        for x in range(0, target_image_width - template_width + 1):
            if min_distance_found <= stopping_threshold:
                break

            for y in range(0, target_image_height - template_height + 1):
                if min_distance_found <= stopping_threshold:
                    break

                top_left = (x, y)
                bottom_right = (x + template_width, y + template_height)

                target_patch_hu_moments = cv2.HuMoments(cv2.moments(target_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]))
                distance = self.__custom_match_shapes(template_hu_moments, target_patch_hu_moments)

                if distance < min_distance_found:
                    match_found = True
                    min_distance_found = distance
                    min_top_left = top_left
                    min_bottom_right = bottom_right

        if render_match and match_found:
            frame_copy = self.frame.copy()
            cv2.rectangle(frame_copy, min_top_left, min_bottom_right, 255, 2)
            self.render_image(frame_copy,
                              'Match top left (%d, %d), bottom right (%d, %d), distance %03.2f' % (min_top_left[0],
                                                                                                   min_top_left[1],
                                                                                                   min_bottom_right[0],
                                                                                                   min_bottom_right[1],
                                                                                                   min_distance_found))

        if match_found and min_distance_found <= threshold:
            return TemplateMatch(min_top_left, min_bottom_right, min_distance_found, threshold)
        else:
            return None

    def find_best_feature_based_match_sift(self,
                                           template_path: str,
                                           threshold: float = 50.0,
                                           template_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                                           frame_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                                           match_horizontal_mirror: bool = False):
        self.__assert_controller_has_frame()

        template = cv2.imread(template_path, cv2.IMREAD_COLOR)

        return self.__find_best_feature_based_match(template,
                                                    cv2.xfeatures2d.SIFT_create(),
                                                    cv2.NORM_L2,
                                                    threshold,
                                                    template_pre_processing_chain,
                                                    frame_pre_processing_chain,
                                                    match_horizontal_mirror)

    def find_best_feature_based_match_surf(self,
                                           template_path: str,
                                           threshold: float = 50.0,
                                           template_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                                           frame_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                                           match_horizontal_mirror: bool = False):
        self.__assert_controller_has_frame()

        template = cv2.imread(template_path, cv2.IMREAD_COLOR)

        return self.__find_best_feature_based_match(template,
                                                    cv2.xfeatures2d.SURF_create(),
                                                    cv2.NORM_L2,
                                                    threshold,
                                                    template_pre_processing_chain,
                                                    frame_pre_processing_chain,
                                                    match_horizontal_mirror)

    def find_best_feature_based_match_orb(self,
                                          template_path: str,
                                          threshold: float = 50.0,
                                          template_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                                          frame_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                                          match_horizontal_mirror: bool = False):
        self.__assert_controller_has_frame()

        template = cv2.imread(template_path, cv2.IMREAD_COLOR)

        return self.__find_best_feature_based_match(template,
                                                    cv2.ORB_create(),
                                                    cv2.NORM_HAMMING,
                                                    threshold,
                                                    template_pre_processing_chain,
                                                    frame_pre_processing_chain,
                                                    match_horizontal_mirror)

    def __find_best_feature_based_match(self,
                                        template,
                                        detector,
                                        distance_measure,
                                        threshold: float,
                                        template_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                                        frame_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                                        match_horizontal_mirror: bool = False):
        target_image = self.frame
        if frame_pre_processing_chain is not None:
            target_image = frame_pre_processing_chain.apply(target_image)

        if template_pre_processing_chain is not None:
            template = template_pre_processing_chain.apply(template)

        key_points_template, descriptors_template = detector.detectAndCompute(template, None)
        key_points_target, descriptors_target = detector.detectAndCompute(target_image, None)

        brute_force_matcher = cv2.BFMatcher(distance_measure, crossCheck=True)
        matches = brute_force_matcher.match(descriptors_template, descriptors_target)

        if match_horizontal_mirror:
            template = image_processing.FlipHorizontal().process(template)
            key_points_template, descriptors_template = detector.detectAndCompute(template, None)
            matches = matches + brute_force_matcher.match(descriptors_template, descriptors_target)

        matches = sorted(matches, key=lambda x: x.distance)

        # matching_result = cv2.drawMatches(template,
        #                                   key_points_template,
        #                                   self.frame,
        #                                   key_points_target,
        #                                   matches[:5],
        #                                   None)
        # cv2.imshow("Matches", matching_result)
        #
        # image = cv2.drawKeypoints(template, key_points_template, None)
        #
        # self.render_image(image)

        if matches and matches[0].distance <= threshold:
            return FeatureBasedMatch((int(key_points_target[matches[0].trainIdx].pt[0]),
                                      int(key_points_target[matches[0].trainIdx].pt[1])),
                                     matches[0].distance,
                                     threshold)
        else:
            return None

    @staticmethod
    def __custom_match_shapes(template_hu_moments, target_patch_hu_moments):
        eps = 1.e-5
        distance = 0.0
        anyA = False
        anyB = False

        for i in range(0, 6):
            abs_moment_a = abs(template_hu_moments[i])
            abs_moment_b = abs(target_patch_hu_moments[i])

            if abs_moment_a > 0:
                anyA = True
            if abs_moment_b > 0:
                anyB = True

            if template_hu_moments[i] > 0:
                sign_moment_a = 1
            elif template_hu_moments[i] < 0:
                sign_moment_a = -1
            else:
                sign_moment_a = 0

            if target_patch_hu_moments[i] > 0:
                sign_moment_b = 1
            elif target_patch_hu_moments[i] < 0:
                sign_moment_b = -1
            else:
                sign_moment_b = 0

            if abs_moment_a > eps and abs_moment_b > eps:
                abs_moment_a = sign_moment_a * np.log10(abs_moment_a)
                abs_moment_b = sign_moment_b * np.log10(abs_moment_b)
                distance += abs(-abs_moment_a + abs_moment_b)

        abs_moment_a = abs(template_hu_moments[6])
        abs_moment_b = abs(target_patch_hu_moments[6])

        if abs_moment_a > 0:
            anyA = True
        if abs_moment_b > 0:
            anyB = True

        if abs_moment_a > eps and abs_moment_b > eps:
            abs_moment_a = np.log10(abs_moment_a)
            abs_moment_b = np.log10(abs_moment_b)
            distance += abs(-abs_moment_a + abs_moment_b)

        if anyA != anyB:
            distance = sys.float_info.max

        return distance

    @staticmethod
    def __has_alpha_channel(image):
        return image.shape[2] > 3

    @staticmethod
    def __split_out_alpha_mask(image):
        if CVController.__has_alpha_channel(image):
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
    def render_image(image, title='Image Display'):
        pyplot.imshow(image, cmap='gray')
        pyplot.title(title), pyplot.xticks([]), pyplot.yticks([])
        pyplot.show()

    @staticmethod
    def is_best_match_the_global_minimum(method: TemplateMatchingMethod):
        # The best match for SQUARE_DIFFERENCE and SQUARE_DIFFERENCE_NORMALIZED is the global minimum value.
        # The best match for the other methods is the global maximum value.
        return method in [TemplateMatchingMethod.SQUARE_DIFFERENCE, TemplateMatchingMethod.SQUARE_DIFFERENCE_NORMALIZED]

    def __match_template(self,
                         template,
                         method: TemplateMatchingMethod = TemplateMatchingMethod.CORRELATION_COEFFICIENT_NORMALIZED,
                         template_pre_processing_chain: image_processing.ImageProcessingStepChain = None,
                         frame_pre_processing_chain: image_processing.ImageProcessingStepChain = None):
        self.__assert_controller_has_frame()

        if self.__has_alpha_channel(template):
            if template_pre_processing_chain is None or not any(isinstance(item, image_processing.ColorTransparentPixels) for item in template_pre_processing_chain.steps):
                template = image_processing.ColorTransparentPixels(image_processing.COLOR_MID_INTENSITY).process(template)

        target_image = self.frame
        if template_pre_processing_chain is not None:
            template = template_pre_processing_chain.apply(template)

        if frame_pre_processing_chain is not None:
            target_image = frame_pre_processing_chain.apply(self.frame)

        return cv2.matchTemplate(target_image, template, method.value)

    def __assert_controller_has_frame(self):
        assert (self.frame is not None), "A frame is required. Use 'capture_frame' or 'load_frame' to prepare frame."

    @staticmethod
    def __convert_rgb_to_bgr(image):
        return image[:, :, ::-1]
