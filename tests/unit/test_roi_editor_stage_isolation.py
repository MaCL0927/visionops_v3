"""Ensure capture ROI and output ROI editors bind to their own visible stages."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HTML = PROJECT_ROOT / "apps/collector_web/frontend/index.html"
VALIDATE_JS = PROJECT_ROOT / "apps/collector_web/frontend/static/js/pages/validate.js"
CAPTURE_JS = PROJECT_ROOT / "apps/collector_web/frontend/static/js/pages/capture.js"


def test_roi_editor_stages_have_unique_ids() -> None:
    html = HTML.read_text(encoding="utf-8")
    assert 'id="capture-roi-stage"' in html
    assert 'id="validate-roi-stage"' in html
    assert html.count('id="capture-roi-stage"') == 1
    assert html.count('id="validate-roi-stage"') == 1


def test_validate_roi_editor_does_not_use_first_generic_stage() -> None:
    source = VALIDATE_JS.read_text(encoding="utf-8")
    assert 'document.getElementById("validate-roi-stage")' in source
    assert 'document.querySelector(".roi-editor-stage")' not in source


def test_capture_roi_editor_uses_its_unique_stage() -> None:
    source = CAPTURE_JS.read_text(encoding="utf-8")
    assert 'document.getElementById("capture-roi-stage")' in source
