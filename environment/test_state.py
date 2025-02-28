from environment.state import GameState, Piece, Player
import pytest

def test_game_state():
    state = GameState()
    assert state.current_player == Player.RED
    assert state.get_open_columns() == [0, 1, 2, 3, 4, 5, 6]
    assert state.board[0].stack == []

    # Add a piece to the 1-idx column
    state.add(0)
    assert state.board[0].stack == [Piece(Player.RED)]
    assert state.current_player == Player.YELLOW
    assert state.get_open_columns() == [0, 1, 2, 3, 4, 5, 6]

    state.add(1)
    assert state.board[1].stack == [Piece(Player.YELLOW)]
    assert state.current_player == Player.RED
    assert state.get_open_columns() == [0, 1, 2, 3, 4, 5, 6]
    assert state.board[2].stack == []

    state.add(2)
    assert state.board[2].stack == [Piece(Player.RED)]
    assert state.board[1].stack == [Piece(Player.YELLOW)]
    assert state.current_player == Player.YELLOW
    assert state.get_open_columns() == [0, 1, 2, 3, 4, 5, 6]
    assert state.board[3].stack == []

    for _ in range(5):
        state.add(1)
    assert state.board[1].stack == [
        Piece(Player.YELLOW),
        Piece(Player.YELLOW), 
        Piece(Player.RED), 
        Piece(Player.YELLOW), 
        Piece(Player.RED),
        Piece(Player.YELLOW),
    ]
    with pytest.raises(ValueError):
        state.add(1)
    
def test_vertical_win():
    state = GameState()
    for row in range(4):
        for col in range(2):
            state.add(col)
            if row == 3 and col == 0:
                assert state.is_game_over()
                assert state.scan_for_winner() == Player.RED
                break
            else:
                assert not state.is_game_over()
                assert state.scan_for_winner() is None
    
def test_horizontal_win():
    state = GameState()
    for col in range(4):
        for row in range(2):
            state.add(col)
            if row == 0 and col == 3:
                assert state.is_game_over()
                assert state.scan_for_winner() == Player.RED
                break
            else:
                assert not state.is_game_over()
                assert state.scan_for_winner() is None

def test_diagonal_win():
    """
          0
        0 1
      0 1 0
    0 1 0 1
    """
    state = GameState()
    for i in range(4):
        for j in range(i):
            state.add(j)
            if i == 3 and j == 3:
                assert state.is_game_over()
                assert state.scan_for_winner() == Player.RED
                break
            else:
                assert not state.is_game_over()
                assert state.scan_for_winner() is None
    
