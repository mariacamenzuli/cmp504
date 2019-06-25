import cmp504
from assertpy import assert_that


def test_find_template_match_square_difference_when_match_exists():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    result = cv_ctrl.find_template_match("cmp504/data/test/wargroove_cherrystone_stronghold.png",
                                         threshold=0.2,
                                         method=cmp504.computer_vision.TemplateMatchingMethod.SQUARE_DIFFERENCE_NORMALIZED)

    assert_that(result).is_not_none()


def test_find_template_match_square_difference_when_match_does_not_exist():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    result = cv_ctrl.find_template_match("cmp504/data/test/discord_icon.png",
                                         threshold=0.2,
                                         method=cmp504.computer_vision.TemplateMatchingMethod.SQUARE_DIFFERENCE_NORMALIZED)

    assert_that(result).is_none()


def test_find_template_match_correlation_coefficient_when_match_exists():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    result = cv_ctrl.find_template_match("cmp504/data/test/wargroove_cherrystone_stronghold.png",
                                         threshold=0.8,
                                         method=cmp504.computer_vision.TemplateMatchingMethod.CORRELATION_COEFFICIENT_NORMALIZED)

    assert_that(result).is_not_none()


def test_find_template_match_correlation_coefficient_when_match_does_not_exist():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    result = cv_ctrl.find_template_match("cmp504/data/test/discord_icon.png",
                                         threshold=0.8,
                                         method=cmp504.computer_vision.TemplateMatchingMethod.CORRELATION_COEFFICIENT_NORMALIZED)

    assert_that(result).is_none()


def test_find_template_match_cross_correlation_when_match_exists():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    result = cv_ctrl.find_template_match("cmp504/data/test/wargroove_cherrystone_stronghold.png",
                                         threshold=0.8,
                                         method=cmp504.computer_vision.TemplateMatchingMethod.CROSS_CORRELATION_NORMALIZED)

    assert_that(result).is_not_none()


def test_find_template_match_cross_correlation_when_match_does_not_exist():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    result = cv_ctrl.find_template_match("cmp504/data/test/discord_icon.png",
                                         threshold=0.8,
                                         method=cmp504.computer_vision.TemplateMatchingMethod.CROSS_CORRELATION_NORMALIZED)

    assert_that(result).is_none()


def test_find_template_match_with_transparent_template_when_match_exists():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_screenshot.png")
    result = cv_ctrl.find_template_match("cmp504/data/test/wargroove_cherrystone_stronghold_with_transparency.png",
                                         threshold=0.8,
                                         method=cmp504.computer_vision.TemplateMatchingMethod.CROSS_CORRELATION_NORMALIZED)

    assert_that(result).is_not_none()


def test_find_text_in_frame():
    cv_ctrl = cmp504.computer_vision.CVController()
    cv_ctrl.load_frame("cmp504/data/test/wargroove_text_panel.png")
    result = cv_ctrl.find_text()

    assert_that(result).contains("Income")
    assert_that(result).contains("Stronghold")

    cv_ctrl.load_frame("cmp504/data/test/numbers.png")
    result = cv_ctrl.find_text()

    assert_that(result).is_equal_to("650 3428")
