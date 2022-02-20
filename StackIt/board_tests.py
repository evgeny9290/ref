import numpy as np


def print_boards_scores(ai, game):
    game.print()
    print('#############begin here###########')
    # print(g.board.current_player)
    move1, score1 = ai.best_move_short(game)
    move2, score2 = ai.best_move_norm(game)
    print('move from short')
    print(f'move: {move1} - score: {score1}')
    game.make_move(*move1)
    game.print()
    game.move_undo()
    print('move from long')
    print(f'move: {move2} - score: {score2}')
    game.make_move(*move2)
    game.print()
    game.move_undo()


def test_boards_one(ai_type, game, depth):
    ai = ai_type(depth)

    game.board.board = np.array([[0,4,0,3],
                              [0,3,0,4],
                              [0,4,4,0]])
    game.board.player = np.array([[0,1,0,1],
                               [0,2,0,1],
                               [0,2,2,0]])
    print_boards_scores(ai, game)


def test_boards_two(ai_type, game, depth):
    ai = ai_type(depth)

    game.board.board = np.array([[4,4,4],
                              [0,4,0],
                              [4,4,4]])
    game.board.player = np.array([[2,1,2],
                               [0,1,0],
                               [2,1,2]])
    print_boards_scores(ai, game)


def test_boards_three(ai_type, game, depth):
    ai = ai_type(depth)

    game.board.board = np.array([
        [0, 4],
        [4, 3]
    ])
    game.board.player = np.array([
        [0, 1],
        [1, 2]
    ])
    print_boards_scores(ai, game)


def test_boards_four(ai_type, game, depth):
    ai = ai_type(depth)

    game.board.board = np.array([[0,4,0,3],
                              [4,4,4,4],
                              [0,4,4,4]])
    game.board.player = np.array([[0,1,0,1],
                               [1,1,1,1],
                               [0,2,1,2]])
    print_boards_scores(ai, game)