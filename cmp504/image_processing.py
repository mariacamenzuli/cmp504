import cv2

COLOR_WHITE = 255
COLOR_MID_INTENSITY = 127


class ImageProcessingStep:
    def process(self, image): pass


class ImageProcessingStepChain:
    def __init__(self, steps: [] = None):
        if steps is None:
            self.steps = []
        else:
            self.steps = steps

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


class BGRA2BGR(ImageProcessingStep):
    def process(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


class CannyEdgeDetection(ImageProcessingStep):
    def process(self, image):
        return cv2.Canny(image, 100, 200)


class GaussianBlur(ImageProcessingStep):
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def process(self, image):
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)


class Threshold(ImageProcessingStep):
    def __init__(self, threshold: int = 127):
        self.threshold = threshold

    def process(self, image):
        return cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)[1]


class AdaptiveThreshold(ImageProcessingStep):
    def process(self, image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


class Invert(ImageProcessingStep):
    def process(self, image):
        return cv2.bitwise_not(image)


class FlipHorizontal(ImageProcessingStep):
    def process(self, image):
        return cv2.flip(image, 1)


class ColorTransparentPixels(ImageProcessingStep):
    def __init__(self, color: int = 127):
        self.color = color

    def process(self, image):
        if image.shape[2] > 3:
            mask = image[:, :, 3] == 0
            image[mask] = [self.color, self.color, self.color, self.color]
            return BGRA2BGR().process(image)
        else:
            return image
