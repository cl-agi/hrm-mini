# 30x30 maze dataset
import json

from datasets import load_dataset

sample = {
    'source': None, 
    'question': ' ## #      #  #   ### #       #  # ##     #  # # # #  #               ### ### #  ##    ##     # #     #  #    ##    # # # #   # #  ##     #   #     #      ## ##   ##  # ## #   ##     # ## # #   #  #   ##  ##    #  ##### #      #     # #     ##   #    #  # ##      # #   ##     # ## ##     ### #         ##      ##  #  ########      #   #     ##  #  #         # # # # ### #  ##      ##   #     ## # ## ### ##    #  #   #   #        #  ## #    #    #    # G# ###   #####   ##  # ##   ##  ##     ## ###   #       ## ##    ### #       ##      ##   #####  # # #   # ##        #  # #   #### #  #   #         #    #    ## #     ### # ##     #    # # ####  ##    #####  #   #    # # #####   #  #     #   ##   ###   # # ###        S  #       #  ##  #  #    # #        # #    #        ## # ##  ###        # ##     ##  #    #    #  #  #   # ## ### ## # #  #      # # #  ##       # #  #      ##    #     # #  # #', 
    'answer': ' ## #      #  #   ### #       #  # ##     #  # # # #  #          ooo  ### ### #  ##    ##     #o#ooo  #  #    ##    # # # # oo# #oo##     #   #     #   ooo## ##oo ##  # ## #   ##    o# ## # #o  #  #   ##  ##    #oo##### #ooooo #     # #     ##o  #    #  #o##      # #   ## ooo # ## ## oo  ### #         ##ooo   ##  #oo########      #   #oooo ##  #o # oooo    # # # # ###o#  ## ooooo##oo #     ## # ##o### ##    #  #o  #   # ooooooo#  ## #    # oo #    # G# ###   #####   ## o# ##   ##  ##     ## ###   # o     ## ##    ### #       ## o    ##   #####  # # #   # ## o      #  # #   #### #  #   # o       #    #    ## #     ###o# ##     #    # # ####  ##   o#####  #   #    # # #####   #o #     #   ##   ###   # # ###o       S  #       #  ##  #  #o   # # oooooo # #    #      oo## # ##  ###oooooo  # ##   oo##  #    #    #  #oo#ooo# ##o### ## # #  #      #o#o#o ##oo     # #  #      ## ooo#ooooo# #  # #', 
    'rating': 112}

def get_dataset():
    return load_dataset("sapientinc/maze-30x30-hard-1k", cache_dir="/dev/shm/datasets", split="test")

def get_prompt():
    return 'Find a shortest path from S to G in the following {height}-row, {width}-column maze.\n\nInput format:\nThe JSON array contains exactly {height} strings, one per row from top to bottom. Each string contains exactly {width} cells from left to right. Every character inside each row string, including every leading or trailing space, is a cell.\n\nPath requirements:\nRow indices run from 1 through {height}, top to bottom; column indices run from 1 through {width}, left to right.\n\'#\' is an impassable wall. A space, \'S\', and \'G\' are traversable cells; \'S\' is the start and \'G\' is the goal.\nEach supplied maze has exactly one S and exactly one G, and a path from S to G exists.\nA valid path is a sequence of cells beginning at S and ending at G. Each move goes to one horizontally or vertically adjacent traversable cell within the grid.\nEvery move costs one step. Path length is the number of moves, equal to the number of cells in the path minus one. Each move is counted exactly once.\nReturn a path whose length is the minimum among all valid S-to-G paths in this maze.\nIf more than one shortest path exists, return any one of them.\nMark exactly the cells on this single path, with each cell visited once.\n\n<maze_rows>\n{question}\n</maze_rows>\n\nOutput cell mapping:\nFor every grid position, use the corresponding output character:\n| Input cell | Output character at the same position |\n|---|---|\n| Wall \'#\' | \'#\' |\n| Start \'S\' | \'S\' |\n| Goal \'G\' | \'G\' |\n| Empty space on the selected path | Lowercase \'o\' |\n| Empty space outside the selected path | Space \' \' |\n\nFinal output format:\nThe entire final response must be one JSON object with exactly one key, "answer".\nThe first non-whitespace character must be \'{{\', and the last non-whitespace character must be \'}}\'.\nAfter JSON decoding, the value of "answer" must be a string of exactly {num_cells} characters: concatenate all {height} solved rows from top to bottom, with the {width} cells of each row read left to right.\nThe string may contain only \'#\', \' \', \'S\', \'G\', and \'o\'. Spaces inside the string are cells, including spaces at row boundaries.\n'

def format_prompt(question: str, height=None, width=None):
    # This module's get_dataset() loads the explicitly named 30x30 dataset.
    # Other dimensions must be supplied together; never guess shape from length.
    if height is None and width is None:
        height, width = 30, 30
    elif height is None or width is None:
        raise ValueError("Provide both maze height and width")
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError("Maze height and width must be positive integers")
    if not isinstance(question, str) or len(question) != height * width:
        raise ValueError("Maze question length must equal height times width")
    encoded_question = json.dumps([question[start:start + width] for start in range(0, len(question), width)], indent=2)
    return get_prompt().format(height=height, width=width, num_cells=height * width, question=encoded_question)

def validate(question: str, answer: str):
    size = 30
    # Check dimensions
    if len(question) != size * size or len(answer) != size * size:
        return False

    # Find S and G positions
    if 'S' not in question or 'G' not in question:
        return False
    s_pos = question.index('S')
    g_pos = question.index('G')

    # Check answer preserves structure
    for i in range(len(question)):
        q, a = question[i], answer[i]
        if q == '#' and a != '#':
            return False
        if q == 'S' and a != 'S':
            return False
        if q == 'G' and a != 'G':
            return False
        if a == 'o' and q != ' ':
            return False
        if q == ' ' and a not in (' ', 'o'):
            return False

    # Collect path cells (S, G, and all 'o')
    path_cells = set()
    for i in range(len(answer)):
        if answer[i] == 'o':
            path_cells.add(i)
    path_cells.add(s_pos)
    path_cells.add(g_pos)

    def neighbors(pos):
        r, c = divmod(pos, size)
        result = []
        if r > 0: result.append(pos - size)
        if r < size - 1: result.append(pos + size)
        if c > 0: result.append(pos - 1)
        if c < size - 1: result.append(pos + 1)
        return result

    # BFS from S through path cells
    visited = set([s_pos])
    queue = [s_pos]
    while queue:
        curr = queue.pop(0)
        for n in neighbors(curr):
            if n in path_cells and n not in visited:
                visited.add(n)
                queue.append(n)

    # G must be reachable and all path cells must be connected
    if g_pos not in visited:
        return False
    if visited != path_cells:
        return False

    return True
