from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from apps.server_api.backend.services.sam_annotation_service import (
    SAM_MODEL_ENV,
    SamAnnotationEngine,
    simplify_closed_polygon,
)


def test_simplify_closed_polygon_reduces_redundant_points() -> None:
    polygon = []
    for x in range(0, 101):
        polygon.append([x, 0])
    for y in range(1, 101):
        polygon.append([100, y])
    for x in range(99, -1, -1):
        polygon.append([x, 100])
    for y in range(99, 0, -1):
        polygon.append([0, y])

    simplified = simplify_closed_polygon(polygon, epsilon=1.0, max_points=32)

    assert 4 <= len(simplified) <= 8
    xs = [point[0] for point in simplified]
    ys = [point[1] for point in simplified]
    assert min(xs) == 0
    assert max(xs) == 100
    assert min(ys) == 0
    assert max(ys) == 100


def test_sam_box_prompt_returns_editable_polygon(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "sam2.1_s.pt"
    model_path.write_bytes(b"placeholder")
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"placeholder")
    monkeypatch.setenv(SAM_MODEL_ENV, str(model_path))

    contour = [
        [10, 10], [30, 10], [50, 10], [70, 10], [90, 10],
        [90, 30], [90, 50], [90, 70], [90, 90],
        [70, 90], [50, 90], [30, 90], [10, 90],
        [10, 70], [10, 50], [10, 30],
    ]

    class FakeModel:
        def predict(self, **kwargs):
            assert kwargs["source"] == str(image_path)
            assert kwargs["bboxes"] == [5.0, 5.0, 95.0, 95.0]
            return [SimpleNamespace(masks=SimpleNamespace(xy=[contour]))]

    engine = SamAnnotationEngine(tmp_path, model_factory=lambda _: FakeModel())
    result = engine.predict_box(image_path, [5, 5, 95, 95], image_size=(100, 100))

    assert result["status"] == "ok"
    assert result["mode"] == "box_prompt"
    assert result["point_count"] >= 4
    assert result["point_count"] < len(contour)
    assert result["mask_area_px"] > 0


def test_sam_box_prompt_rejects_tiny_box(tmp_path: Path) -> None:
    engine = SamAnnotationEngine(tmp_path, model_factory=lambda _: object())
    try:
        engine.predict_box(tmp_path / "image.jpg", [1, 1, 2, 2], image_size=(100, 100))
    except ValueError as exc:
        assert "过小" in str(exc)
    else:
        raise AssertionError("tiny SAM prompt box should fail")


def test_quick_segmentation_command_preserves_nested_masks(tmp_path: Path, monkeypatch) -> None:
    import zipfile

    from apps.server_api.backend.services.annotation_service import AnnotationService
    from apps.server_api.backend.services.ingest_service import BatchService

    upload = tmp_path / "seg.zip"
    with zipfile.ZipFile(upload, "w") as archive:
        for index in range(5):
            archive.writestr(f"images/{index}.jpg", b"fake-image")
            # Both classes are nested in the same images; two instances per class
            # make the default 10-instance threshold pass.
            rows = []
            for _ in range(2):
                rows.append("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9")
                rows.append("1 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7")
            archive.writestr(f"labels/{index}.txt", "\n".join(rows) + "\n")

    batches = BatchService(tmp_path / "batches", ("segmentation",))
    batch = batches.create_from_zip(upload, device_id="server", task_type="segmentation")
    service = AnnotationService(batches, tmp_path / "server_data")
    captured: dict[str, str] = {}

    def fake_run_shell(cmd, log_file, cwd):
        captured["cmd"] = cmd
        weights = service.quick_root(batch["batch_id"]) / "runs" / "segment_quick" / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"fake")

    monkeypatch.setattr(service, "_run_shell", fake_run_shell)
    result = service._quick_train_worker(
        log_file=SimpleNamespace(write=lambda *_: None, flush=lambda: None),
        update=lambda *_: None,
        batch_id=batch["batch_id"],
        task="segmentation",
        classes=["foam_ring", "ring_mouth"],
        project_root=tmp_path,
    )

    assert result["train_images"] == 5
    assert "overlap_mask=False" in captured["cmd"]
    assert "mask_ratio=2" in captured["cmd"]
    assert "degrees=12.0" in captured["cmd"]
    assert "amp=True" in captured["cmd"]
