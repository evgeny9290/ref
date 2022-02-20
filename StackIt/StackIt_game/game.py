from copy import deepcopy
from .Constants import *
from .board import Board


class Game:
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
        self.board = Board(rows=self.rows, cols=self.cols)

    def change_player(self):
        if self.board.current_player == 1:
            return 2
        if self.board.current_player == 2:
            return 1

    def print(self):
        self.board.print()

    def score(self, player):
        boxes = 0
        for x in range(self.board.board.shape[0]):
            for y in range(self.board.board.shape[1]):
                if self.board.player[x, y] == player:
                    boxes += self.board.board[x, y]
        return boxes
        # player_1_boxes = np.sum(self.board.player == 1)
        # player_2_boxes = np.sum(self.board.player == 2)
        # return np.subtract(player_2_boxes, player_1_boxes)

    def make_move(self, x, y, amount=1):
        if self.board.is_legal(x, y):
            self.board.history.append((self.board.current_player,
                                       deepcopy(self.board.board),
                                       deepcopy(self.board.player)))
            self.move(x, y, amount)
        else:
            raise Exception('cannot move ontop of other player')

    def move(self, x, y, amount=1):
        if amount != 1:
            self.board.board[x, y] += amount
        else:
            self.board.board[x, y] += 1

        self.board.player[x, y] = self.board.current_player
        if self.board.board[x, y] == 5:
            self.board.throw_over(x, y)
            to_throw = self.board.places_to_throw()
            while to_throw:
                for x, y in to_throw:
                    self.board.throw_over(x, y)
                to_throw = self.board.places_to_throw()

        self.board.current_player = self.change_player()

    def simmulate_game_move(self, x, y):
        self.make_move(x, y)
        return self.board.board, self.board.player

    def deepcopy(self):
        return deepcopy(self.board.board), deepcopy(self.board.player)

    def ai_move(self, board):
        self.board = board
        self.change_player()

    def move_undo(self):
        curr_player, board, player = self.board.history.pop()
        self.board.current_player = curr_player
        self.board.board = board
        self.board.player = player

    @classmethod
    def from_existing_board(cls, board, player, current_player=1):
        import copy
        instance = cls(board.shape[0], board.shape[1])
        instance.board.board = copy.deepcopy(board)
        instance.board.player = copy.deepcopy(player)
        instance.board.current_player = current_player
        return instance

def test_to_throw(g):
    # g = Game(3)
    g.print()
    g.make_move(1, 1, 4)
    g.print()
    g.make_move(1, 2, 4)
    g.print()
    g.make_move(0, 1, 4)
    g.print()
    g.make_move(2, 0, 4)
    g.print()
    g.make_move(1, 1)
    g.print()
    g.make_move(2, 0)
    g.print()
    # print('aaaaaaaaaaaaaaaa')
    # print(g.board.board[2,0])

if __name__ == '__main__':
    g = Game(3)
    test_to_throw(g)
    # hist = g.board.history
    # for player, board, player_board in hist:
    #     print(player)
    #     print(board)
    #     print(player_board)