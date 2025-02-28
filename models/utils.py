from environment.state import GameState, Piece, Player
import torch 
from torch import nn
from torch.nn import functional as F

EMPTY = -1

def get_stack_tensor(stack: list[Piece]) -> torch.Tensor:
    """
    Get the stack as a tensor of shape (6, 1)
    """
    return torch.tensor([piece.player.value for piece in stack], dtype=torch.float32)

def fill_board(game: GameState, player: Player) -> torch.Tensor:
    """
    Fill the board with the current player's move
    """
    board = game.board
    board_tensor = torch.zeros(6, 7)
    for col in range(7):
        stack_tensor = get_stack_tensor(board[col].stack)
        # Flip the bits if player.value == 1
        if player.value == 1:
            stack_tensor = 1 - stack_tensor
        board_tensor[:len(board[col].stack), col] = stack_tensor
        board_tensor[len(board[col].stack):, col] = EMPTY
    return board_tensor

def get_open_columns(board: torch.Tensor) -> torch.Tensor:
    """
    Get the open columns of the board
    """
    return (board == EMPTY).sum(dim=-2) > 0
