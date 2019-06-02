import cmp504
import numpy as np
from assertpy import assert_that


def test_find_match_when_match_exists():
    computer_vision_ctrl = cmp504.computer_vision.CVController()
    computer_vision_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    match = computer_vision_ctrl.find_match("cmp504/data/test/wargroove_cherrystone_stronghold.png")

    assert_that(np.shape(match)[1]).is_equal_to(1)


# def test_capture_frame():
#     computer_vision_ctrl = cmp504.computer_vision.CVController()
#     computer_vision_ctrl.capture_frame()
#     match = computer_vision_ctrl.find_match("cmp504/data/test/wargroove_cherrystone_stronghold.png")
#
#     assert_that(np.shape(match)[1]).is_equal_to(1)
