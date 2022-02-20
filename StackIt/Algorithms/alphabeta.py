from collections import defaultdict


class AlphaBeta:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.moves = defaultdict(int)

    def best_move_norm(self, game):  # removed player
        move, score = self.alphabeta_long(game, self.max_depth, float('-inf'), float('inf'), True)
        return move, score

    def best_move_short(self, game):
        move, score = self.alphabeta_short(game, self.max_depth, float('-inf'), float('inf'))
        return move, score

    def alphabeta_short(self, game, depth, alpha, beta):  # negamax
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
            _, score = self.alphabeta_short(game, depth-1, -beta, -alpha)
            score *= -1
            # print('depth', depth, 'score', score)
            # game.print()
            game.move_undo()
            if score > best_score:
                best_score = score
                action = move
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return action, best_score

    def alphabeta_long(self, game, depth, alpha, beta, max_player):
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
                _, score = self.alphabeta_long(game, depth - 1, alpha, beta, False)
                # print('depth', depth, 'score', score)
                # game.print()
                # print(score)
                game.move_undo()
                if best_score < score:
                    best_score = score
                    action = move
                if best_score >= beta:
                    break
                alpha = max(alpha, best_score)
            return action, best_score
        else:
            best_score = float('inf')
            action = None
            for move in game.board.get_legal_moves():
                game.make_move(*move)
                self.moves['total_moves'] += 1
                _, score = self.alphabeta_long(game, depth - 1, alpha, beta, True)
                # print('depth', depth, 'score', score)
                # game.print()
                # print(score)
                game.move_undo()
                if best_score > score:
                    best_score = score
                    action = move
                if alpha >= best_score:
                    break
                beta = min(beta, best_score)
            return action, best_score
