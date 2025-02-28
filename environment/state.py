from dataclasses import dataclass
from enum import Enum


NUM_COLS = 7
NUM_ROWS = 6

class Player(Enum):
    RED = 0
    YELLOW = 1

class Winner(Enum):
    RED = 0
    YELLOW = 1
    DRAW = 2


@dataclass
class Piece:
    player: Player

class Column:
    def __init__(self, stack: list[Piece] | None = None):
        self.stack: list[Piece] = stack if stack is not None else []

    def add(self, player: Player):
        if len(self.stack) == NUM_ROWS:
            raise ValueError("Column is full")
        self.stack.append(Piece(player))
    
    def is_full(self):
        return len(self.stack) == NUM_ROWS

class GameState:
    def __init__(self, board: list[Column] | None = None, current_player: Player | None = None):
        if board is None:
            self.board = [Column() for _ in range(NUM_COLS)]
        else:
            self.board = board
        self.current_player = current_player if current_player is not None else Player.RED
        
    def add(self, col: int):
        self.board[col].add(self.current_player)
        self.current_player = (
            Player.RED if self.current_player == Player.YELLOW 
            else Player.YELLOW
        )

    def is_game_over(self) -> bool:
        return all(col.is_full() for col in self.board) or self.scan_for_winner() is not None

    def get_winner(self) -> Winner | None:
        winner = self.scan_for_winner()
        if winner is None:
            return None
        elif all(col.is_full() for col in self.board):
            return Winner.DRAW
        else:
            return Winner.RED if winner == Player.RED else Winner.YELLOW

    def get_open_columns(self) -> list[int]:
        return [i for i, col in enumerate(self.board) if not col.is_full()]

    def scan_for_winner(self) -> Player | None:
        # Check all possible winning alignments
        horizontal_winner = self._check_horizontal()
        if horizontal_winner:
            return horizontal_winner
            
        vertical_winner = self._check_vertical()
        if vertical_winner:
            return vertical_winner
            
        diagonal_winner = self._check_diagonal()
        if diagonal_winner:
            return diagonal_winner
            
        return None
    
    def _check_horizontal(self) -> Player | None:
        """Check for 4 consecutive pieces horizontally."""
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS - 3):
                # Skip if not enough pieces in this column at this row
                if row >= len(self.board[col].stack):
                    continue
                    
                player = self.board[col].stack[row].player
                if (row < len(self.board[col+1].stack) and player == self.board[col+1].stack[row].player and
                    row < len(self.board[col+2].stack) and player == self.board[col+2].stack[row].player and
                    row < len(self.board[col+3].stack) and player == self.board[col+3].stack[row].player):
                    return player
        return None
    
    def _check_vertical(self) -> Player | None:
        """Check for 4 consecutive pieces vertically."""
        for col in range(NUM_COLS):
            column = self.board[col].stack
            for row in range(len(column) - 3):
                player = column[row].player
                if (player == column[row+1].player and
                    player == column[row+2].player and
                    player == column[row+3].player):
                    return player
        return None
    
    def _check_diagonal(self) -> Player | None:
        """Check for 4 consecutive pieces diagonally."""
        # Check diagonals going up-right
        for row in range(3, NUM_ROWS):
            for col in range(NUM_COLS - 3):
                # Skip if not enough pieces in this column at this row
                if row >= len(self.board[col].stack):
                    continue
                    
                player = self.board[col].stack[row].player
                if (row-1 < len(self.board[col+1].stack) and player == self.board[col+1].stack[row-1].player and
                    row-2 < len(self.board[col+2].stack) and player == self.board[col+2].stack[row-2].player and
                    row-3 < len(self.board[col+3].stack) and player == self.board[col+3].stack[row-3].player):
                    return player
        
        # Check diagonals going down-right
        for row in range(NUM_ROWS - 3):
            for col in range(NUM_COLS - 3):
                # Skip if not enough pieces in this column at this row
                if row >= len(self.board[col].stack):
                    continue
                    
                player = self.board[col].stack[row].player
                if (row+1 < len(self.board[col+1].stack) and player == self.board[col+1].stack[row+1].player and
                    row+2 < len(self.board[col+2].stack) and player == self.board[col+2].stack[row+2].player and
                    row+3 < len(self.board[col+3].stack) and player == self.board[col+3].stack[row+3].player):
                    return player
        
        return None
    
    
    