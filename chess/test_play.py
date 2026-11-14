import chess

from play import model_turn, move_prompt


class ScriptedPlayer:
    """Returns the given moves in order, like a model would."""

    def __init__(self, moves):
        self.moves = list(moves)

    def sample_move(self, prompt):
        return self.moves.pop(0)


def test_move_prompt():
    assert move_prompt([]) == ";1."
    assert move_prompt(["e4"]) == ";1.e4 "
    assert move_prompt(["e4", "e5"]) == ";1.e4 e5 2."


def test_model_turn_legal_first_try():
    move, illegal = model_turn(ScriptedPlayer(["e4"]), chess.Board(), [], retries=5)
    assert move == chess.Move.from_uci("e2e4") and illegal == []


def test_model_turn_retries_then_plays():
    move, illegal = model_turn(ScriptedPlayer(["Ke2", "--", "", "Nf3"]), chess.Board(), [], retries=5)
    assert move == chess.Move.from_uci("g1f3")
    assert illegal == ["Ke2", "--", ""]


def test_model_turn_forfeits_after_retries():
    move, illegal = model_turn(ScriptedPlayer(["Qh5"] * 3), chess.Board(), [], retries=2)
    assert move is None and len(illegal) == 3


def test_legal_move():
    from play import legal_move
    board = chess.Board()
    assert legal_move(board, "e4") == chess.Move.from_uci("e2e4")
    for bad in ["Ke2", "--", "", "e5", "xyz"]:
        assert legal_move(board, bad) is None


def test_elo_from_score():
    from play import elo_from_score
    assert elo_from_score(1500, 0.5) == 1500
    assert round(elo_from_score(1500, 0.75)) == 1691  # +400*log10(3)
    assert elo_from_score(1500, 0.0) is None and elo_from_score(1500, 1.0) is None


def test_combined_elo_matches_single_level_and_handles_extremes():
    from play import combined_elo, elo_from_score
    rating, error = combined_elo([(1500, 100, 0.75)])
    assert abs(rating - elo_from_score(1500, 0.75)) < 0.01 and 35 < error < 45  # 400/ln10/sqrt(100*0.1875)
    assert combined_elo([(1320, 50, 0.0), (1500, 50, 0.0)]) == (None, None)
    two, _ = combined_elo([(1320, 100, 0.6), (1700, 100, 0.2)])
    assert 1320 < two < 1700


def test_opponents_set_every_option():
    from play import opponents
    assert opponents([0], None) == [("skill 0", "skill", 0, {"UCI_LimitStrength": False, "Skill Level": 0})]
    label, kind, value, options = opponents([0], [1500])[0]
    assert (label, kind, value) == ("elo 1500", "elo", 1500) and options["UCI_LimitStrength"] is True


def test_parse_clock():
    import pytest
    from play import parse_clock
    assert parse_clock("120+1") == (120.0, 1.0)
    assert parse_clock("60") == (60.0, 0.0)
    with pytest.raises(ValueError):
        parse_clock("fast")
