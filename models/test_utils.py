from environment.state import GameState, NUM_ROWS, NUM_COLS
from models.utils import fill_board, EMPTY
import torch

def test_fill_board():
    game = GameState()
    
    board_tensor = fill_board(game, game.current_player)
    assert board_tensor.shape == (NUM_ROWS, NUM_COLS)
    assert torch.all(board_tensor == EMPTY)

    game.add(0)
    board_tensor = fill_board(game, game.current_player)
    assert board_tensor.shape == (NUM_ROWS, NUM_COLS)
    assert (board_tensor == EMPTY).sum() == NUM_ROWS * NUM_COLS - 1
    assert board_tensor[0, 0] == 1

    game.add(1)
    board_tensor = fill_board(game, game.current_player)
    assert board_tensor.shape == (NUM_ROWS, NUM_COLS)
    assert (board_tensor == EMPTY).sum() == NUM_ROWS * NUM_COLS - 2
    assert board_tensor[0, 0] == 0
    assert board_tensor[0, 1] == 1

