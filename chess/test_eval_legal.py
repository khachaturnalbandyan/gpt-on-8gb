import chess

from eval_legal import phase_of, pick_positions


def test_phase_of():
    assert [phase_of(p) for p in (0, 19, 20, 59, 60, 300)] == ["opening"] * 2 + ["middle"] * 2 + ["late"] * 2


def test_pick_positions_replays_the_game():
    game = ["e4", "e5", "Nf3", "Nc6", "Bb5"]
    (moves, board), = pick_positions([game], seed=1)
    assert moves == game[: len(moves)]
    expected = chess.Board()
    for san in moves:
        expected.push_san(san)
    assert board.fen() == expected.fen()
