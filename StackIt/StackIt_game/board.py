import numpy as np
from .Constants import *
from copy import deepcopy


class Board:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.board, self.player = self._init()
        self.current_player = 1
        self.history = []

    def _init(self):
        return np.zeros(shape=(self.rows, self.cols), dtype=int), np.zeros(shape=(self.rows, self.cols), dtype=int)

    def places_to_throw(self):
        to_throw = []
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                if self.board[row, col] >= 5:
                    to_throw.append((row, col))
        return to_throw

    def throw_over(self, x, y):
        self.board[x, y] -= 4

        if self.in_bounds(x, y - 1):  # up
            if self.player[x, y - 1] == self.current_player:
                self.board[x, y - 1] += 1
            else:
                self.player[x, y - 1] = self.current_player
                self.board[x, y - 1] = 1

        if self.in_bounds(x, y + 1):  # down
            if self.player[x, y + 1] == self.current_player:
                self.board[x, y + 1] += 1
            else:
                self.player[x, y + 1] = self.current_player
                self.board[x, y + 1] = 1

        if self.in_bounds(x - 1, y):  # left
            if self.player[x - 1, y] == self.current_player:
                self.board[x - 1, y] += 1
            else:
                self.player[x - 1, y] = self.current_player
                self.board[x - 1, y] = 1
        if self.in_bounds(x + 1, y):  # right
            if self.player[x + 1, y] == self.current_player:
                self.board[x + 1, y] += 1
            else:
                self.player[x + 1, y] = self.current_player
                self.board[x + 1, y] = 1

    def in_bounds(self, x, y):
        return 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]

    def check_legal_move(self, x, y, player):
        if self.in_bounds(x, y):
            if self.player[x, y] == player or self.player[x, y] == 0:
                return True
        return False

    def get_legal_moves(self):
        players_pos = list(zip(*np.where(self.player == self.current_player)))
        empty_pos = list(zip(*np.where(self.player == 0)))
        return players_pos + empty_pos

    def is_legal(self, x, y):
        if (self.player[x, y] == 0 or self.player[x, y] == self.current_player) and self.in_bounds(x, y):
            return True
        return False

    @staticmethod
    def _player_color(player):
        if player == 1:
            return PLAYER1COLOR
        elif player == 2:
            return PLAYER2COLOR
        else:
            return EMPTY

    def check_win(self):
        player_1_win = np.sum(self.player == 1)
        if player_1_win == self.player.shape[0] * self.player.shape[1]:
            return 1

        player_2_win = np.sum(self.player == 2)
        if player_2_win == self.player.shape[0] * self.player.shape[1]:
            return 2

        return 0

    def print(self):
        current_color = self._player_color(self.current_player)
        spacing = "  "
        print(f"Current board at move {len(self.history)} for player "
              f"{current_color + str(self.current_player) + EMPTY}:")
        print()
        print(spacing, ' ', ' '.join([str(x) for x in range(self.cols)]))
        print(spacing, ' ', ' '.join(['-' for x in range(self.cols)]))
        for y, row in enumerate(self.board):
            print(spacing + str(y) + '|', end=' ')
            for x, field in enumerate(row):
                color = self._player_color(self.player[y, x])
                print(color + str(field) + EMPTY, end=' ')
            print()
        print()

    def make_move(self, x, y, amount=1):
        if self.is_legal(x, y):
            self.history.append((self.current_player, deepcopy(self.board), deepcopy(self.player)))
            self.move(x, y, amount)
        else:
            raise Exception('cannot move ontop of other player')

    def change_player(self):
        if self.current_player == 1:
            return 2
        if self.current_player == 2:
            return 1

    def move(self, x, y, amount=1):
        if amount != 1:
            self.board[x, y] += amount
        else:
            self.board[x, y] += 1

        self.player[x, y] = self.board.current_player
        if self.board[x, y] == 5:
            self.throw_over(x, y)
            to_throw = self.places_to_throw()
            while to_throw:
                for x, y in to_throw:
                    self.throw_over(x, y)
                to_throw = self.places_to_throw()

        self.board.current_player = self.change_player()


if __name__ == '__main__':
    b = Board(3,3)
    b.print()
    b.make_move(2,2)
    b.print()
    b.make_move(2,1)
    b.print()
    # b.player = np.ones((3,3)) -1
    # # b.player[1,1] = 2
    # # b.board[1,1]=3
    # # b.player[2,2] = 2
    # # b.board[2,2]=4
    # print(b.get_legal_moves(1))
    # b.print()
    # print(b.check_win())

