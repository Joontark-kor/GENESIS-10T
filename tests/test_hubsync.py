from genesis.hubsync import Uploader


def test_uploader_skips_empty_folder_and_counts_pushes(tmp_path):
    calls = []
    up = Uploader("u/r", tmp_path / "run", "tok", every=999, push=lambda *a, **k: calls.append(a[:2]))
    assert up.push_once() is False and calls == []
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "a.txt").write_text("x")
    assert up.push_once() is True and up.pushes == 1 and calls == [("u/r", tmp_path / "run")]


def test_uploader_survives_a_failed_push_and_never_keeps_the_message(tmp_path):
    (tmp_path / "f").write_text("x")
    def boom(*a, **k):
        raise RuntimeError("Authorization: Bearer SECRET")
    up = Uploader("u/r", tmp_path, "tok", push=boom)
    assert up.push_once() is False and up.failures == 1
    assert up.last_error == "RuntimeError" and "SECRET" not in up.last_error


def test_stop_does_a_final_push(tmp_path):
    (tmp_path / "f").write_text("x")
    calls = []
    up = Uploader("u/r", tmp_path, "tok", every=999, push=lambda *a, **k: calls.append(1))
    up.stop()
    assert calls == [1]


def test_get_secret_falls_back_to_environment_and_names_the_miss(monkeypatch):
    import pytest
    from genesis.hubsync import get_secret
    monkeypatch.setenv("GENESIS_TEST_SECRET", "abc")
    assert get_secret("GENESIS_TEST_SECRET") == "abc"
    monkeypatch.delenv("GENESIS_TEST_SECRET")
    with pytest.raises(RuntimeError) as err:
        get_secret("GENESIS_TEST_SECRET")
    assert "GENESIS_TEST_SECRET" in str(err.value) and "abc" not in str(err.value)
