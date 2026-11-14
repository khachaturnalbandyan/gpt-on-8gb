import pytest

from elo_report import build_report


def game(level, result, color="white", reason="checkmate", kind="elo"):
    return {"checkpoint": "c.pt", "opponent": kind, kind: level, "result": result, "model_color": color,
            "reason": reason, "plies": 40, "sf_clock": "120+1", "sf_time": None}


def test_merges_levels_and_estimates_elo():
    games = [game(1500, "win"), game(1500, "loss", "black"), game(1500, "draw"), game(1500, "draw", "black"),
             game(1320, "loss", reason="forfeit_illegal")]
    report = build_report(games)
    assert report["games"] == 5
    assert report["per_level"]["elo 1500"]["score"] == 0.5
    assert report["per_level"]["elo 1500"]["elo_estimate"] == 1500
    assert report["per_level"]["elo 1500"]["as_black"] == {"win": 0, "draw": 1, "loss": 1}
    assert report["per_level"]["elo 1320"]["forfeits"] == 1
    assert report["elo_estimate"] is not None and report["elo_estimate"] < 1500


def test_zero_score_gives_no_estimate():
    report = build_report([game(1320, "loss"), game(1320, "loss", "black")])
    assert report["elo_estimate"] is None and "note" in report


def test_skill_files_have_no_elo():
    report = build_report([game(0, "win", kind="skill")])
    assert "elo_estimate" not in report and report["per_level"]["skill 0"]["score"] == 1.0


def test_rejects_mixed_files():
    with pytest.raises(ValueError):
        build_report([game(1320, "loss"), game(0, "loss", kind="skill")])
