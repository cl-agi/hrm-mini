import torch


def valid_sudoku(pred, clues):
    """Native Sudoku tokens: BOS then 81 digits, 0 denotes a blank."""
    board = pred[:, 1:].reshape(-1, 9, 9)
    digits = torch.arange(1, 10, device=pred.device)
    rows = (board.sort(dim=-1).values == digits).all(dim=-1).all(dim=-1)
    cols = (board.transpose(1, 2).sort(dim=-1).values == digits).all(dim=-1).all(dim=-1)
    boxes = board.reshape(-1, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    boxes = (boxes.sort(dim=-1).values == digits).all(dim=-1).all(dim=-1)
    givens = ((clues[:, 1:] == 0) | (pred[:, 1:] == clues[:, 1:])).all(dim=-1)
    return rows & cols & boxes & givens


if __name__ == '__main__':
    grid = torch.tensor([[(r*3+r//3+c)%9+1 for c in range(9)] for r in range(9)])
    good = torch.cat([torch.tensor([0]), grid.flatten()])[None]
    empty = torch.zeros_like(good)
    assert valid_sudoku(good, empty).item()
    assert valid_sudoku(good, good).item()
    other = good.clone(); other[:,1:] = other[:,1:] % 9 + 1
    assert valid_sudoku(other, empty).item()
    assert not valid_sudoku(other, good).item()
    bad = good.clone(); bad[:,1] = 0
    assert not valid_sudoku(bad, empty).item()
    latin = torch.tensor([[(r+c)%9+1 for c in range(9)] for r in range(9)])
    bad = torch.cat([torch.tensor([0]),latin.flatten()])[None]
    assert not valid_sudoku(bad, empty).item()
    bad = good.clone(); bad[:,[1,2]] = bad[:,[2,1]]
    assert not valid_sudoku(bad, empty).item()
    print('PASS: valid alternatives, givens, blanks, columns, boxes')
