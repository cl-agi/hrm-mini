"""Within-box Sudoku simulated annealing used for E31 search comparisons.

Lewis-inspired protocol: 1000 unconditional calibration swaps, temperature
std(energy) floored at .01, chain length blanks squared, cooling .99, and a
fresh box-valid restart after 20 chains without accepted downhill moves.
The solver receives clues only, never the reference solution.
"""
import numpy as np
try:
    from numba import njit
except ImportError:
    raise ImportError('E31 annealing requires numba: pip install -e ".[search]"') from None

@njit
def energy(b):
    cost = 0
    for i in range(9):
        rows = np.zeros(10, np.int64)
        cols = np.zeros(10, np.int64)
        for j in range(9):
            rows[b[9*i+j]] = 1
            cols[b[9*j+i]] = 1
        cost += 18 - rows[1:].sum() - cols[1:].sum()
    return cost

@njit
def initialize(puzzle):
    b = puzzle.copy()
    for box in range(9):
        cells = np.array([(box//3*3+r)*9+box%3*3+c for r in range(3) for c in range(3)])
        used = np.zeros(10, np.int64)
        for cell in cells:
            used[b[cell]] = 1
        missing = np.where(used[1:] == 0)[0] + 1
        np.random.shuffle(missing)
        k = 0
        for cell in cells:
            if b[cell] == 0:
                b[cell] = missing[k]
                k += 1
    return b

@njit
def pair(editable, partners, counts):
    i = editable[np.random.randint(len(editable))]
    j = partners[i, np.random.randint(counts[i])]
    return i, j

@njit
def solve(puzzle, seed, budget, stride=100):
    np.random.seed(seed)
    counts = np.zeros(81, np.int64)
    partners = np.zeros((81, 8), np.int64)
    for i in range(81):
        for j in range(81):
            if puzzle[i] == 0 and puzzle[j] == 0 and i != j and i//27 == j//27 and i%9//3 == j%9//3:
                partners[i, counts[i]] = j
                counts[i] += 1
    editable = np.where(counts > 0)[0]
    b = initialize(puzzle)
    if energy(b) == 0:
        return b, np.zeros((1, 5)), b.reshape(1,81), 0, 0, 0, 0, 0.0
    if len(editable) == 0:
        raise ValueError('No legal swap and board is unsolved')
    calibration = np.empty(1000)
    for k in range(1000):
        i,j = pair(editable, partners, counts)
        b[i],b[j] = b[j],b[i]
        calibration[k] = energy(b)
    t0 = max(float(np.std(calibration)), 0.01)
    b = initialize(puzzle)
    e = energy(b)
    temp = t0
    ml = int(np.sum(puzzle == 0))**2
    # Columns: proposals, current energy, actual temperature, restart count, event.
    capacity = budget//stride + budget//ml + 10
    trace = np.empty((capacity,5))
    boards = np.empty((capacity,81), np.int64)
    trace[0] = np.array([0., e, temp, 0., 0.])
    boards[0] = b
    n = 1
    stagnant = 0
    downhill = False
    reheats = 0
    accepted = 0
    uphill = 0
    step = 0
    for step in range(1,budget+1):
        i,j = pair(editable, partners, counts)
        b[i],b[j] = b[j],b[i]
        proposed = energy(b)
        delta = proposed - e
        if delta <= 0 or np.random.random() < np.exp(-delta/temp):
            e = proposed
            accepted += 1
            uphill += int(delta > 0)
            downhill = downhill or delta < 0
        else:
            b[i],b[j] = b[j],b[i]
        if step % stride == 0 or e == 0:
            trace[n] = np.array([float(step), e, temp, float(reheats), 0.])
            boards[n] = b
            n += 1
        if e == 0:
            break
        if step % ml == 0:
            stagnant = 0 if downhill else stagnant + 1
            downhill = False
            temp *= .99
            if stagnant >= 20:
                b = initialize(puzzle)
                e = energy(b)
                temp = t0
                stagnant = 0
                reheats += 1
                trace[n] = np.array([float(step), e, temp, float(reheats), 1.])
                boards[n] = b
                n += 1
    if trace[n-1,0] != step:
        trace[n] = np.array([float(step), e, temp, float(reheats), 0.])
        boards[n] = b
        n += 1
    return b, trace[:n], boards[:n], step, reheats, accepted, uphill, t0

def valid(b, puzzle):
    grid = b.reshape(9,9)
    goal = set(range(1,10))
    return bool(np.all(b[puzzle != 0] == puzzle[puzzle != 0]) and
                all(set(row) == goal for row in grid) and
                all(set(col) == goal for col in grid.T) and
                all(set(grid[r:r+3,c:c+3].ravel()) == goal for r in (0,3,6) for c in (0,3,6)))

