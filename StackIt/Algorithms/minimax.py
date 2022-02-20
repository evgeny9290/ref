from collections import defaultdict


class MiniMax:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.moves = defaultdict(int)

    def best_move_norm(self, game):
        move, score = self.minimax_normal(game, self.max_depth, True)
        return move, score

    def best_move_short(self, game):
        move, score = self.minimax_shortened(game, self.max_depth)
        return move, score

    def minimax_shortened(self, game, depth):  # negamax
        if depth == 0:
            return None, game.score(1) - game.score(2)
        if game.board.check_win():
            return None, -100000

        best_score = float('-inf')
        action = None
        for move in game.board.get_legal_moves():
            game.make_move(*move)
            self.moves['total_moves'] += 1
            # print('depth', depth, 'move', move)
            _, score = self.minimax_shortened(game, depth-1)
            score *= -1
            # print('depth', depth, 'score', score)
            # game.print()
            game.move_undo()
            if score > best_score:
                best_score = score
                action = move
        return action, best_score

    def minimax_normal(self, game, depth, max_player):
        if depth == 0:
            return None, game.score(1) - game.score(2)
        if game.board.check_win():
            return None, 100000

        if max_player:
            best_score = float('-inf')
            action = None
            for move in game.board.get_legal_moves():
                game.make_move(*move)
                self.moves['total_moves'] += 1
                _, score = self.minimax_normal(game, depth - 1, False)
                # print('depth', depth, 'score', score)
                # game.print()
                # print(score)
                game.move_undo()
                if best_score < score:
                    best_score = score
                    action = move
            return action, best_score
        else:
            best_score = float('inf')
            action = None
            for move in game.board.get_legal_moves():
                game.make_move(*move)
                self.moves['total_moves'] += 1
                _, score = self.minimax_normal(game, depth - 1, True)
                # print('depth', depth, 'score', score)
                # game.print()
                # print(score)
                game.move_undo()
                if best_score > score:
                    best_score = score
                    action = move
            return action, best_score


    # def simmulate_move(self, move, game):
    #     game.make_move(*move)
    #     return game.board.board, game.board.player
    #
    # def generate_successors(self, action, game):
    #     game_board, player_board = self.simmulate_move(action, game)
    #     moves = []
    #     valid_moves = game.board.get_legal_moves()
    #     for move in valid_moves:
    #         temp_board = deepcopy(game_board)
    #         temp_player_board = deepcopy(player_board)
    #         game.board.board = temp_board
    #         game.board.player = temp_player_board
    #         new_board = self.simmulate_move(move, game)
    #         moves.append(new_board)
    #
    #     return moves

