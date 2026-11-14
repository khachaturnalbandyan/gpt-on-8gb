import numpy as np

from prepare_chess import (LOOKUP, VOCAB, estimated_seconds, format_game, iter_games, parse_moves,
                           reject_reason)

GOOD_HEADERS = {"WhiteElo": "1900", "BlackElo": "1850", "TimeControl": "300+0", "Termination": "Normal"}
TWENTY_PLIES = ["e4", "e5"] * 10


def test_vocab_is_32_unique_chars():
    assert len(VOCAB) == 32 == len(set(VOCAB))


def test_parse_moves_strips_numbers_comments_nags_and_result():
    text = "1. e4 { [%eval 0.2] } 1... e5?! 2. Nf3 $1 Nc6 3.Bb5 a6!? 4. O-O e8=Q+ 1/2-1/2"
    assert parse_moves(text) == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "O-O", "e8=Q+"]


def test_format_game():
    assert format_game(["e4", "e5", "Nf3"]) == ";1.e4 e5 2.Nf3"


def test_estimated_seconds():
    assert estimated_seconds("300+3") == 420
    assert estimated_seconds("-") is None
    assert estimated_seconds("") is None


def test_reject_reason():
    assert reject_reason(GOOD_HEADERS, TWENTY_PLIES) is None
    assert reject_reason({**GOOD_HEADERS, "BlackElo": "1799"}, TWENTY_PLIES) == "low_elo"
    assert reject_reason({**GOOD_HEADERS, "BlackElo": "?"}, TWENTY_PLIES) == "no_elo"
    assert reject_reason({**GOOD_HEADERS, "TimeControl": "60+0"}, TWENTY_PLIES) == "bullet_or_no_clock"
    assert reject_reason({**GOOD_HEADERS, "Termination": "Abandoned"}, TWENTY_PLIES) == "bad_termination"
    assert reject_reason(GOOD_HEADERS, TWENTY_PLIES[:-1]) == "too_short"


def test_iter_games_splits_games():
    pgn = ['[Event "a"]', '[Result "0-1"]', "", " 0-1", "", '[Event "b"]', "", "1. e4 e5", "2. Nf3 1-0", ""]
    games = list(iter_games(pgn))
    assert [h["Event"] for h, _ in games] == ["a", "b"]
    assert games[1][1] == "1. e4 e5 2. Nf3 1-0"


def test_lookup_round_trip():
    text = ";1.e4 e5 2.Nf3 O-O#"
    ids = LOOKUP[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
    assert "".join(VOCAB[i] for i in ids) == text
