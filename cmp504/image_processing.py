import cv2


class ImageProcessingStep:
    def process(self, image): pass


class ImageProcessingStepChain:
    def __init__(self):
        self.steps = []

    def append(self, step: ImageProcessingStep):
        self.steps.append(step)

    def apply(self, image):
        processed_image = image

        for step in self.steps:
            processed_image = step.process(processed_image)

        return processed_image


class Resize(ImageProcessingStep):
    def __init__(self, scale_factor_x: float, scale_factor_y: float):
        self.scale_factor_x = scale_factor_x
        self.scale_factor_y = scale_factor_y

    def process(self, image):
        return cv2.resize(image,
                          None,
                          fx=self.scale_factor_x,
                          fy=self.scale_factor_y,
                          interpolation=cv2.INTER_CUBIC)


class BGR2Grayscale(ImageProcessingStep):
    def process(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class CannyEdgeDetection(ImageProcessingStep):
    def process(self, image):
        return cv2.Canny(image, 100, 200)


class GaussianBlur(ImageProcessingStep):
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def process(self, image):
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)


class Threshold(ImageProcessingStep):
    def process(self, image):
        # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


class Invert(ImageProcessingStep):
    def process(self, image):
        return cv2.bitwise_not(image)
