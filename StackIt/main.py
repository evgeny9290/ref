from StackIt_game.game import Game
from Algorithms.minimax import MiniMax
from Algorithms.alphabeta import AlphaBeta
from board_tests import test_boards_one, test_boards_two, test_boards_three, test_boards_four
import time
import numpy as np


def play_game(game, ai, depth):
    PLAYERS = {1: 'human', 2: ai(depth)}
    commands = {'e': 'exist', 'p': 'pass', 'move x y': '(x,y)'}
    while True:
        if game.board.check_win():
            print(f'player {game.board.check_win()} has won the game')
            break
        print(game.board.current_player)
        if PLAYERS[game.board.current_player] == 'human':
            print('available commands:')
            for short, comm in commands.items():
                print(f'{short} for {comm}')
            game.board.print()
            command = input('please input a move to play ').strip()

            if command == 'e':
                break
            elif command == 'p':
                game.board.current_player = game.change_player()
                continue
            elif command.startswith('move'):
                command, x, y = command.split(' ')
                try:
                    game.make_move(int(x), int(y))
                    game.print()
                except Exception:
                    print('move out of bounds')
        else:
            move, score = PLAYERS[game.board.current_player].best_move_norm(game)
            print(f'Algo move: {move} - Algo score: {score}')
            game.make_move(*move)


def performance_test_short(game, depth, ai_type):
    for max_depth in range(depth + 1):
        ai = ai_type(max_depth)
        time_short = time.time()
        move_short, score_short = ai.best_move_short(game)
        print(f"depth: {max_depth} - move: {move_short} - score: {score_short} - time: {time.time() - time_short}"
              f" - moves made: {ai.moves['total_moves']}")


def performance_test_long(game, depth, ai_type):
    for max_depth in range(depth + 1):
        ai = ai_type(max_depth)
        time_norm = time.time()
        move_norm, score_norm = ai.best_move_norm(game)
        print(f"depth: {max_depth} - move: {move_norm} - score: {score_norm} - time: {time.time() - time_norm}"
              f" - moves made: {ai.moves['total_moves']}")


def board_tests(ai, game, depth):
    test_boards_one(ai, game, depth)
    test_boards_two(ai, game, depth)
    test_boards_three(ai, game, depth)
    test_boards_four(ai, game, depth)


if __name__ == '__main__':
    g1 = Game(2, 2)
    g2 = Game(3, 3)
    g3 = Game(2, 3)
    # ai = AlphaBeta
    # play_game(g3, ai, 12)
    g1.board.board = np.array([[0,4,0,3],
                              [0,3,0,4],
                              [0,4,4,0]])
    g1.board.player = np.array([[0,1,0,1],
                               [0,2,0,1],
                               [0,2,2,0]])
    print('short func test minimax')
    performance_test_short(g1, 6, MiniMax)
    print()
    print('long func test minimax')
    performance_test_long(g1, 6, MiniMax)
    print()
    print('#############################################')
    print()
    print('short func test alphabeta')
    performance_test_short(g1, 10, AlphaBeta)
    print()
    print('long func test alphabeta')
    performance_test_long(g1, 10, AlphaBeta)
    print(end='\n\n')
    print('############ board tests minimax ############')
    board_tests(MiniMax, g1, 6)
    print('############ board tests  alphabeta############')
    board_tests(AlphaBeta, g1, 10)
