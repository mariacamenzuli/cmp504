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
