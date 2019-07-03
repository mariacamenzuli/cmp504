import cmp504
from assertpy import assert_that


def test_find_template_match_square_difference_when_match_exists():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    match = cv_ctrl.find_template_match("cmp504/data/test/wargroove_cherrystone_stronghold.png",
                                        threshold=0.2,
                                        method=cmp504.computer_vision.TemplateMatchingMethod.SQUARE_DIFFERENCE_NORMALIZED)

    assert_that(match).is_not_none()
    assert_that(match.top_left).is_equal_to((247, 305))
    assert_that(match.bottom_right).is_equal_to((311, 391))
    assert_that(match.mid_point).is_equal_to((279, 348))


def test_find_template_match_square_difference_when_match_does_not_exist():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    match = cv_ctrl.find_template_match("cmp504/data/test/discord_icon.png",
                                        threshold=0.2,
                                        method=cmp504.computer_vision.TemplateMatchingMethod.SQUARE_DIFFERENCE_NORMALIZED)

    assert_that(match).is_none()


def test_find_template_match_correlation_coefficient_when_match_exists():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    match = cv_ctrl.find_template_match("cmp504/data/test/wargroove_cherrystone_stronghold.png",
                                         threshold=0.8,
                                         method=cmp504.computer_vision.TemplateMatchingMethod.CORRELATION_COEFFICIENT_NORMALIZED)

    assert_that(match).is_not_none()
    assert_that(match.top_left).is_equal_to((247, 305))
    assert_that(match.bottom_right).is_equal_to((311, 391))
    assert_that(match.mid_point).is_equal_to((279, 348))


def test_find_template_match_correlation_coefficient_when_match_does_not_exist():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    match = cv_ctrl.find_template_match("cmp504/data/test/discord_icon.png",
                                        threshold=0.8,
                                        method=cmp504.computer_vision.TemplateMatchingMethod.CORRELATION_COEFFICIENT_NORMALIZED)

    assert_that(match).is_none()


def test_find_template_match_cross_correlation_when_match_exists():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    match = cv_ctrl.find_template_match("cmp504/data/test/wargroove_cherrystone_stronghold.png",
                                        threshold=0.8,
                                        method=cmp504.computer_vision.TemplateMatchingMethod.CROSS_CORRELATION_NORMALIZED)

    assert_that(match).is_not_none()
    assert_that(match.top_left).is_equal_to((247, 305))
    assert_that(match.bottom_right).is_equal_to((311, 391))
    assert_that(match.mid_point).is_equal_to((279, 348))


def test_find_template_match_cross_correlation_when_match_does_not_exist():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    match = cv_ctrl.find_template_match("cmp504/data/test/discord_icon.png",
                                        threshold=0.8,
                                        method=cmp504.computer_vision.TemplateMatchingMethod.CROSS_CORRELATION_NORMALIZED)

    assert_that(match).is_none()


def test_find_template_match_with_transparent_template_when_match_exists():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    match = cv_ctrl.find_template_match("cmp504/data/test/wargroove_cherrystone_stronghold_with_transparency.png",
                                        threshold=0.8,
                                        method=cmp504.computer_vision.TemplateMatchingMethod.CROSS_CORRELATION_NORMALIZED)

    assert_that(match).is_not_none()
    assert_that(match.top_left).is_equal_to((247, 305))
    assert_that(match.bottom_right).is_equal_to((311, 391))
    assert_that(match.mid_point).is_equal_to((279, 348))


def test_find_template_matches():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/mario_screenshot.png")
    matches = cv_ctrl.find_template_matches("cmp504/data/test/mario_coin.png",
                                            threshold=0.98,
                                            method=cmp504.computer_vision.TemplateMatchingMethod.CROSS_CORRELATION_NORMALIZED)

    assert_that(matches).is_not_empty()
    assert_that(matches).contains_only(cmp504.computer_vision.TemplateMatch((73, 82), (82, 96)),
                                       cmp504.computer_vision.TemplateMatch((87, 82), (96, 96)),
                                       cmp504.computer_vision.TemplateMatch((101, 82), (110, 96)),
                                       cmp504.computer_vision.TemplateMatch((115, 82), (124, 96)),
                                       cmp504.computer_vision.TemplateMatch((129, 82), (138, 96)),
                                       cmp504.computer_vision.TemplateMatch((59, 114), (68, 128)),
                                       cmp504.computer_vision.TemplateMatch((73, 114), (82, 128)),
                                       cmp504.computer_vision.TemplateMatch((87, 114), (96, 128)),
                                       cmp504.computer_vision.TemplateMatch((101, 114), (110, 128)),
                                       cmp504.computer_vision.TemplateMatch((115, 114), (124, 128)),
                                       cmp504.computer_vision.TemplateMatch((129, 114), (138, 128)),
                                       cmp504.computer_vision.TemplateMatch((143, 114), (152, 128)),
                                       cmp504.computer_vision.TemplateMatch((59, 146), (68, 160)),
                                       cmp504.computer_vision.TemplateMatch((73, 146), (82, 160)),
                                       cmp504.computer_vision.TemplateMatch((87, 146), (96, 160)),
                                       cmp504.computer_vision.TemplateMatch((101, 146), (110, 160)),
                                       cmp504.computer_vision.TemplateMatch((115, 146), (124, 160)),
                                       cmp504.computer_vision.TemplateMatch((129, 146), (138, 160)),
                                       cmp504.computer_vision.TemplateMatch((143, 146), (152, 160)))


def test_find_text_in_frame_with_no_preprocessing():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_text_panel.png")
    result = cv_ctrl.find_text()

    assert_that(result).contains("Income")
    assert_that(result).contains("Stronghold")

    cv_ctrl.load_frame("cmp504/data/test/numbers.png")
    result = cv_ctrl.find_text()

    assert_that(result).is_equal_to("650 3428")


def test_find_text_in_frame_with_preprocessing():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_tile_info_panel.png")
    pre_processing_steps = cmp504.image_processing.ImageProcessingStepChain()
    pre_processing_steps.append(cmp504.image_processing.BGR2Grayscale())
    pre_processing_steps.append(cmp504.image_processing.Resize(3, 3))
    result = cv_ctrl.find_text(pre_processing_chain=pre_processing_steps)

    assert_that(result).contains("Income")
    assert_that(result).contains("Stronghold")


def test_something():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot_mirrored_commander.png")
    match = cv_ctrl.find_template_match_hu_moments("cmp504/data/test/wargroove_commander_portrait.png",
                                                   render_match=True)

    assert_that(match).is_not_none()
    # assert_that(match.top_left).is_equal_to((247, 305))
    # assert_that(match.bottom_right).is_equal_to((311, 391))
    # assert_that(match.mid_point).is_equal_to((279, 348))
