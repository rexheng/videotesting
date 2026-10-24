import numpy as np

from streetseg.models import YoloDetector, YoloSemanticSegmenter


def test_missing_weights_are_not_downloaded(monkeypatch, tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("runtime download")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    missing = tmp_path / "yolo26n-sem.pt"
    try:
        YoloSemanticSegmenter(missing).predict(frame)
    except FileNotFoundError as exc:
        assert "not downloaded at runtime" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
    try:
        YoloDetector(tmp_path / "yolo26n.pt").predict(frame)
    except FileNotFoundError as exc:
        assert "python -m streetseg.download" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
