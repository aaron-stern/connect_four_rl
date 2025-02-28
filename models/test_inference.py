from models.inference import ConnectFourModel
from environment.state import GameState, Player
from models.utils import fill_board
import torch

def test_model():
    model = ConnectFourModel()
    x = torch.randn(1, 1, 6, 7)
    assert model(x).shape == (1, 7)

def test_masking():
    model = ConnectFourModel()
    game = GameState()
    for _ in range(6):
        game.add(0)

    batch = fill_board(game, Player.RED).unsqueeze(0).unsqueeze(0)
    assert model(batch).shape == (1, 7)
    assert torch.all(model(batch)[:, 0] == -torch.inf)
    assert torch.all(model(batch)[:, 1] != -torch.inf)
