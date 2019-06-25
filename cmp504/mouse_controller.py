import time
from pynput.mouse import Button, Controller as MouseControl


class MouseController:
    def __init__(self):
        self.mouse = MouseControl()

    def move_mouse(self, x: int, y: int):
        self.mouse.position = (x, y)

    def move_mouse_smoothly(self, x, y):
        def set_mouse_position(x, y):
            self.mouse.position = (int(x), int(y))

        def smooth_mouse_movement(from_x, from_y, to_x, to_y, speed=0.2):
            steps = 40
            sleep_per_step = speed // steps
            x_delta = (to_x - from_x) / steps
            y_delta = (to_y - from_y) / steps
            # is there a predefined LERP somewhere?
            for step in range(steps):
                new_x = x_delta * (step + 1) + from_x
                new_y = y_delta * (step + 1) + from_y
                set_mouse_position(new_x, new_y)
                time.sleep(sleep_per_step)

        return smooth_mouse_movement(
            self.mouse.position[0],
            self.mouse.position[1],
            x,
            y
        )

    def left_mouse_click(self):
        self.mouse.click(Button.left)

    def right_mouse_click(self):
        self.mouse.click(Button.right)
