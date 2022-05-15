import pygame
import sys
import time
from copy import deepcopy
import random


class Piece:
    """
        Class to memorize a piece on the board
        type: the type of the piece (can be normal piece, king, mounted king)
        position: coordinates of the piece
        captured: boolean value only for king pieces (if true the king is captured)
    """
    def __init__(self, type=None, position=None, captured=False):
        self.type = type
        self.position = position
        if type in {'R', 'B', 'U', 'V'}:
            self.captured = captured

    def change_king(self):
        """
            function that mounts/unmounts the king
        """
        if self.type == 'B':
            return 'U'
        elif self.type == 'U':
            return 'B'
        elif self.type == 'R':
            return 'V'
        elif self.type == 'V':
            return 'R'
        return ''

    def __eq__(self, piece):
        if not isinstance(piece, Piece):
            return False
        if self.type != self.type:
            return False
        if self.position != piece.position:
            return False
        return True

    def __repr__(self):
        return self.type + " " + str(self.position)


class Board:
    """
        Class to memorize a board configuration
        reds: vector of red piece
        blues: vector of blue piece
        red_king / blue_king: king pieces
        size: size of board
        corners: color of corners
        free_space: how a free space is represented
        p_min_tep/p_max_tep: true if a player teleported
    """

    def __init__(self, size=7, free_space='#', p_max_tep=False, p_min_tep=False, orig=None):
        if orig is None:
            self.reds = []
            self.blues = []
            self.corners = ["white", "white", "white", "white"]
            self.size = size
            self.free_space = free_space
            self.p_max_tep = p_max_tep
            self.p_min_tep = p_min_tep

            for i in range(size):
                if i == size // 2:
                    self.red_king = Piece('R', (0, i))
                    self.blue_king = Piece('B', (size - 1, i))
                else:
                    self.reds.append(Piece('r', (0, i)))
                    self.blues.append(Piece('b', (size - 1, i)))
        else:
            self.reds = [piece for piece in orig.reds]
            self.blues = [piece for piece in orig.blues]
            self.blue_king = Piece(orig.blue_king.type, orig.blue_king.position)
            self.red_king = Piece(orig.red_king.type, orig.red_king.position)
            self.corners = [corner for corner in orig.corners]
            self.size = orig.size
            self.free_space = orig.free_space
            self.p_max_tep = orig.p_max_tep
            self.p_min_tep = orig.p_min_tep

    @staticmethod
    def color_corner(king, size):
        """
            Args: king piece and the size of the board
            Returns: position of the corner and the color if the king is in the corner
                     -1 and white if the king is not on the corner
        """
        position = king.position
        type = king.type

        if type in {'b', 'B', 'U'}:
            color = "blue"
        else:
            color = "red"
        if position == (0, 0):
            return 0, color
        if position == (0, size - 1):
            return 1, color
        if position == (size - 1, 0):
            return 2, color
        if position == (size - 1, size - 1):
            return 3, color
        return -1, "white"

    @staticmethod
    def aligned(piece1, piece2, matrix):
        """
            Args: 2 pieces and the board as a matrix
            Check if the 2 pieces are aligned (attacking each other)
            Returns: True if the pieces are aligned / False otherwise
        """
        x1, y1 = piece1.position
        x2, y2 = piece2.position

        directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        for direction in directions:
            nx = x1 + direction[0]
            ny = y1 + direction[1]
            if nx == x2 and ny == y2:
                return True
            while Game.check_coordinates(nx, ny) and matrix[nx][ny] == '#':
                nx += direction[0]
                ny += direction[1]
                if nx == x2 and ny == y2:
                    return True

        return False

    @staticmethod
    def count_attacks(enemy_piece, pieces, matrix):
        """
            Args: an enemy piece, a vector of player pieces and the board as a matrix
            Returns: the number of pieces that attack the enemy piece
        """
        ct = len([piece for piece in pieces if Board.aligned(enemy_piece, piece, matrix)])
        return ct

    def get_matrix(self):
        """
            Function that transforms a Board object
            into a matrix using the codification
        """

        matrix = [[self.free_space for _ in range(self.size)] for _ in range(self.size)]
        for piece in self.blues:
            matrix[piece.position[0]][piece.position[1]] = piece.type
        for piece in self.reds:
            matrix[piece.position[0]][piece.position[1]] = piece.type

        if self.blue_king:
            matrix[self.blue_king.position[0]][self.blue_king.position[1]] = self.blue_king.type

        if self.red_king:
            matrix[self.red_king.position[0]][self.red_king.position[1]] = self.red_king.type

        return matrix

    def get_pieces(self, player):
        """
            Args: name of the player
            Returns: all the pieces and the corners of the player
        """
        if player == "blue":
            return self.blue_king, self.blues, self.corners
        return self.red_king, self.reds, self.corners

    def set_pieces(self, player, pieces, king, corners):
        """
            This functions sets the player's pieces and corners
        """
        if player == "blue":
            self.blue_king = king
            self.blues = pieces
        else:
            self.red_king = king
            self.reds = pieces
        self.corners = corners

    def capture_king(self, player):
        """
            Args: name of the player that captures the king
            The function captures the enemy king setting its captured value to True
            Returns: the new state of the board
        """
        if player == 'blue':
            if self.red_king.type == 'R':
                self.red_king.captured = True
        elif player == 'red':
            if self.blue_king.type == 'B':
                self.blue_king.captured = True
        return self

    def __eq__(self, board):
        return self.get_matrix() == board.get_matrix()

    def __str__(self):
        matrix = self.get_matrix()
        if matrix is None:
            return ""
        txt = ["".join(row) for row in matrix]
        return "\n".join(txt)

    def __repr__(self):
        return str(self)


class Game:
    """
        Game is the class that represents the game itself
        P_MIN - the player (human)
        P_MAX - is the computer
        FREE - the characters that marks an empty cell on the board
        SIZE - number of cells in a row / column
        display - the graphic window
        cells - vector of squares to draw on display
        k_blue, k_red, r_img, b_img - images
        cell_size - size of the cell
        padding - space between the board and the margins of the display
        board - current state of the game
    """

    P_MIN = None
    P_MAX = None
    FREE = '#'
    SIZE = None
    display = None
    cells = None
    k_blue = k_red = r_img = b_img = cell_size = padding = None

    def __init__(self, board=None, size=None):
        self.last_move = None

        if board:
            self.board = board
        else:
            self.board = Board(size, self.__class__.FREE)

            if size is not None:
                self.__class__.SIZE = size

    def draw_grid(self):
        """
            function that draws the grid on the display
        """
        for idx in range(self.__class__.SIZE ** 2):
            row = idx // self.__class__.SIZE
            col = idx % self.__class__.SIZE
            if row == 0 and col == 0:
                if self.board.corners[0] == 'blue':
                    color = (0, 0, 255, 80)
                elif self.board.corners[0] == 'red':
                    color = (255, 0, 0, 80)
                else:
                    color = (255, 255, 255)
            elif row == 0 and col == Game.SIZE - 1:
                if self.board.corners[1] == 'blue':
                    color = (0, 0, 255, 80)
                elif self.board.corners[1] == 'red':
                    color = (255, 0, 0, 80)
                else:
                    color = (255, 255, 255)
            elif row == Game.SIZE - 1 and col == 0:
                if self.board.corners[2] == 'blue':
                    color = (0, 0, 255, 80)
                elif self.board.corners[2] == 'red':
                    color = (255, 0, 0, 80)
                else:
                    color = (255, 255, 255)
            elif row == Game.SIZE - 1 and col == Game.SIZE - 1:
                if self.board.corners[3] == 'blue':
                    color = (0, 0, 255, 80)
                elif self.board.corners[3] == 'red':
                    color = (255, 0, 0, 80)
                else:
                    color = (255, 255, 255)
            else:
                color = (255, 255, 255)
            coords = (col * (self.__class__.cell_size + 1) + self.__class__.padding,
                      row * (self.__class__.cell_size + 1) + self.__class__.padding)
            pygame.draw.rect(self.__class__.display, color, self.__class__.cells[idx])
            matrix = self.board.get_matrix()
            if matrix[row][col] == 'b':
                self.__class__.display.blit(self.__class__.b_img, coords)
            elif matrix[row][col] == 'r':
                self.__class__.display.blit(self.__class__.r_img, coords)
            elif matrix[row][col] == 'R' and not self.board.red_king.captured:
                self.__class__.display.blit(self.__class__.k_red, coords)
            elif matrix[row][col] == 'B' and not self.board.blue_king.captured:
                self.__class__.display.blit(self.__class__.k_blue, coords)
            elif matrix[row][col] == 'U' and not self.board.blue_king.captured:
                self.__class__.display.blit(self.__class__.b_img, coords)
                self.__class__.display.blit(self.__class__.k_blue, coords)
            elif matrix[row][col] == 'V' and not self.board.red_king.captured:
                self.__class__.display.blit(self.__class__.r_img, coords)
                self.__class__.display.blit(self.__class__.k_red, coords)
        pygame.display.update()

    @classmethod
    def opposite_player(cls, player):
        """
            function that switches the player
        """
        return cls.P_MAX if player == cls.P_MIN else cls.P_MIN

    @classmethod
    def get_start_state(cls, display, size=7, cell_size=100, padding=32):
        """
            function that generates the start state
        """
        cls.display = display
        cls.cell_size = cell_size
        cls.padding = padding
        cls.b_img = pygame.image.load('blue.png')
        cls.b_img = pygame.transform.scale(cls.b_img, (cell_size - 10, cell_size - 10))
        cls.r_img = pygame.image.load('red.png')
        cls.r_img = pygame.transform.scale(cls.r_img, (cell_size - 10, cell_size - 10))
        cls.k_blue = pygame.image.load('kblue.png')
        cls.k_blue = pygame.transform.scale(cls.k_blue, (cell_size * 0.75, cell_size * 0.9))
        cls.k_red = pygame.image.load('kred.png')
        cls.k_red = pygame.transform.scale(cls.k_red, (cell_size * 0.75, cell_size * 0.9))
        cls.cells = []
        for row in range(size):
            for col in range(size):
                rect = pygame.Rect(col * (cell_size + 1) + 30, row * (cell_size + 1) + 30, cell_size, cell_size)
                cls.cells.append(rect)

    @classmethod
    def check_coordinates(cls, x, y):
        """
            Utility function to check if the (x, y) coordinates are on board
        """
        if 0 <= x < cls.SIZE and 0 <= y < cls.SIZE:
            return True
        return False

    def final_state(self):
        """"
            Function that checks if the board reached a final state -> the game ends
        """
        if self.board.red_king.captured or len(self.board.reds) == 1:
            return "blue"

        if self.board.blue_king.captured or len(self.board.blues) == 1:
            return "red"

        blue_corners = red_corners = 0
        for corner in self.board.corners:
            if corner == "blue":
                blue_corners += 1
            if corner == "red":
                red_corners += 1
        if blue_corners == 4:
            return "blue"
        if red_corners == 4:
            return "red"

        return False

    def possible_moves(self, player):
        """
            Function that generates all the possible moves of a player
            Returns: (initial_piece, piece_after_move, new_board)
        """

        directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        next_moves = []

        king, pieces, corners = self.board.get_pieces(player)
        matrix = self.board.get_matrix()

        # move the king normally
        cx, cy = king.position
        for direction in directions:
            nx = cx + direction[0]
            ny = cy + direction[1]
            if self.check_coordinates(nx, ny):
                if matrix[nx][ny] == '#':
                    if king.type == 'R' or king.type == 'B':
                        new_king = Piece(king.type, (nx, ny), king.captured)
                    else:
                        new_king = Piece(king.change_king(), (nx, ny), king.captured)
                    corner_idx, color = Board.color_corner(new_king, self.__class__.SIZE)
                    new_corners = [color for color in corners]
                    if corner_idx >= 0:
                        new_corners[corner_idx] = color
                    new_board = Board(orig=self.board)
                    if player == self.__class__.P_MAX:
                        new_board.P_MAX_tep = False
                    else:
                        new_board.p_min_tep = False
                    new_board.set_pieces(player, pieces, new_king, new_corners)
                    next_moves.append((king, new_king, new_board))
                elif king.type == (matrix[nx][ny]).upper():
                    if king.type == 'R' or king.type == 'B':
                        new_king = Piece(king.change_king(), (nx, ny), king.captured)
                    else:
                        new_king = Piece(king.type, (nx, ny), king.captured)
                    corner_idx, color = Board.color_corner(new_king, self.__class__.SIZE)
                    new_corners = [color for color in corners]
                    if corner_idx >= 0:
                        new_corners[corner_idx] = color
                    new_board = Board(orig=self.board)
                    if player == self.__class__.P_MAX:
                        new_board.P_MAX_tep = False
                    else:
                        new_board.p_min_tep = False
                    new_board.set_pieces(player, pieces, new_king, new_corners)
                    next_moves.append((king, new_king, new_board))

        # if mounted
        if king.type == 'V' or king.type == 'U':
            # teleport the king
            teleported = self.board.p_max_tep if player == self.__class__.P_MAX else self.board.p_min_tep
            if not teleported:  # teleport the king
                for piece in pieces:
                    if piece.position != king.position:
                        new_king = Piece(king.type, piece.position, king.captured)
                        corner_idx, color = Board.color_corner(new_king, self.__class__.SIZE)
                        new_corners = [color for color in corners]
                        if corner_idx >= 0:
                            new_corners[corner_idx] = color
                        new_board = Board(orig=self.board)
                        if player == self.__class__.P_MAX:
                            new_board.P_MAX_tep = True
                        else:
                            new_board.p_min_tep = True
                        new_board.set_pieces(player, pieces, new_king, new_corners)
                        next_moves.append((king, new_king, new_board))

        # try to move the pieces
        for piece in pieces:
            if piece.position != king.position:  # if the piece is not mounted
                # try to capture the opposite king
                if king.type == 'B' or king.type == 'U':
                    enemy_king = self.board.red_king
                else:
                    enemy_king = self.board.blue_king

                if Board.aligned(piece, enemy_king, matrix) and enemy_king.type not in {'U', 'V'}:
                    new_pieces = [x for x in pieces if x != piece]
                    new_piece = Piece(piece.type, enemy_king.position)
                    new_pieces.append(new_piece)
                    new_board = Board(orig=self.board)
                    if player == self.__class__.P_MAX:
                        new_board.P_MAX_tep = False
                    else:
                        new_board.p_min_tep = False
                    new_board.set_pieces(player, new_pieces, king, corners)
                    new_board = new_board.capture_king(player)
                    next_moves.append((piece, new_piece, new_board))

                # move the piece normally
                px, py = piece.position
                for direction in directions:
                    nx = px + direction[0]
                    ny = py + direction[1]
                    if self.check_coordinates(nx, ny) and matrix[nx][ny] == '#':
                        new_pieces = [x for x in pieces if x != piece]
                        new_piece = Piece(piece.type, (nx, ny))
                        new_pieces.append(new_piece)
                        new_board = Board(orig=self.board)
                        if player == self.__class__.P_MAX:
                            new_board.P_MAX_tep = False
                        else:
                            new_board.p_min_tep = False
                        new_board.set_pieces(player, new_pieces, king, corners)

                        # check if enemy pieces will be killed
                        new_matrix = new_board.get_matrix()
                        if king.type == 'B' or king.type == 'U':
                            enemy_pieces = [x for x in self.board.reds if Board.aligned(x, new_piece, new_matrix)]
                            for enemy_piece in enemy_pieces:
                                if Board.count_attacks(enemy_piece, new_pieces,  new_matrix) >= 3:
                                    new_board.reds.remove(enemy_piece)
                        else:
                            enemy_pieces = [x for x in self.board.blues if Board.aligned(x, new_piece, new_matrix)]
                            for enemy_piece in enemy_pieces:
                                if Board.count_attacks(enemy_piece, new_pieces, new_matrix) >= 3:
                                    new_board.blues.remove(enemy_piece)
                        next_moves.append((piece, new_piece, new_board))
        return next_moves

    def score1(self):
        """
            function that calculates the score:
            every captured normal piece = +3
            every corner colored = +10
            every attack on the king = +5

            if the enemy does one of the above tasks -> the scores are negative
        """
        s = 0
        # add points for the corners
        king, pieces, corners = self.board.get_pieces(self.__class__.P_MAX)
        enemy_king, enemy_pieces, _ = self.board.get_pieces(self.__class__.P_MIN)

        if king is None:
            return -1000000000
        if enemy_king is None:
            return 1000000000

        for corner in corners:
            if corner == self.__class__.P_MAX:
                s += 10
            elif corner == self.__class__.P_MIN:
                s -= 10

        # add the number of pieces
        s += len(pieces) - len(enemy_pieces)

        # add the number of attacks on the kings
        matrix = self.board.get_matrix()
        s = 5 * Board.count_attacks(enemy_king, pieces, matrix) - 5 * Board.count_attacks(king, enemy_pieces, matrix)
        return s

    def score2(self):
        """
             Function that calculates the score
             score1 +
             for every almost captured enemy piece = +2
             if the pieces advance = +1

             if the player pieces are almost captured -2
        """
        s = self.score1()
        king, pieces, corners = self.board.get_pieces(self.__class__.P_MAX)
        enemy_king, enemy_pieces, _ = self.board.get_pieces(self.__class__.P_MIN)
        matrix = self.board.get_matrix()

        if king is None or enemy_king is None:
            return s

        for piece in pieces:
            if abs(piece.position[0] - (Game.SIZE // 2)):
                s -= 1
            if Board.count_attacks(piece, enemy_pieces, matrix) == 2:
                s -= 2

        for enemy_piece in enemy_pieces:
            if abs(enemy_piece.position[0] - (Game.SIZE // 2)):
                s += 1
            if Board.count_attacks(enemy_piece, enemy_pieces, matrix) == 2:
                s += 2

        return s

    def estimate_score(self, h):
        """
            function that calculates the score
        """
        inf = 1000000000
        winner = self.final_state()
        if winner == self.__class__.P_MAX:
            return inf + h
        elif winner == self.__class__.P_MIN:
            return - inf - h
        else:
            # return self.score1()
            return self.score2()

    def __repr__(self):
        return str(self.board)


class State:

    def __init__(self, game, current_player, depth, parent=None, score=None):
        # board is an instance of game class
        self.game = game
        self.current_player = current_player
        self.depth = depth
        self.parent = parent
        self.score = score
        self.next_moves = []
        self.next_state = None

    def possible_moves(self):
        l_moves = [Game(board, Game.SIZE) for _, _, board in self.game.possible_moves(self.current_player)]
        enemy = Game.opposite_player(self.current_player)
        l_states = [State(move, enemy, self.depth - 1, parent=self) for move in l_moves]

        return l_states

    def __str__(self):
        sir = str(self.game)
        return sir

    def __repr__(self):
        sir = str(self)
        return sir


def min_max(state):
    if state.depth == 0 or state.game.final_state():
        state.score = state.game.estimate_score(state.depth)
        return state

    state.next_moves = state.possible_moves()

    moves_with_scores = [min_max(move) for move in state.next_moves]

    if state.current_player == Game.P_MAX:
        state.next_state = max(moves_with_scores, key=lambda x: x.score)
    else:
        state.next_state = min(moves_with_scores, key=lambda x: x.score)
    state.score = state.next_state.score
    return state


def alpha_beta(alpha, beta, state):
    if state.depth == 0 or state.game.final_state():
        state.score = state.game.estimate_score(state.depth)
        return state

    if alpha > beta:
        return stare

    state.next_moves = state.possible_moves()

    if state.current_player == Game.P_MAX:
        current_score = float('-inf')

        for move in state.next_moves:
            new_state = alpha_beta(alpha, beta, move)
            if current_score < new_state.score:
                state.next_state = new_state
                current_score = new_state.score
            if alpha < new_state.score:
                alpha = new_state.score
                if alpha >= beta:
                    break

    elif state.current_player == Game.P_MIN:
        current_score = float('inf')

        for move in state.next_moves:
            new_state = alpha_beta(alpha, beta, move)
            if current_score > new_state.score:
                state.next_state = new_state
                current_score = new_state.score

            if beta > new_state.score:
                beta = new_state.score
                if alpha >= beta:
                    break
    state.score = state.next_state.score

    return state


class Button:
    def __init__(self, display=None, left=0, top=0, w=0, h=0, background_color=(53, 80, 115),
                 selection_color=(89, 134, 194), text="", font="arial", font_size=16, text_color=(255, 255, 255),
                 value=""):
        self.display = display
        self.background_color = background_color
        self.selection_color = selection_color
        self.text = text
        self.font = font
        self.left = left
        self.top = top
        self.w = w
        self.h = h
        self.selected = False
        self.font_size = font_size
        self.text_color = text_color

        font_obj = pygame.font.SysFont(self.font, self.font_size)
        self.rendered_text = font_obj.render(self.text, True, self.text_color)
        self.rectangle = pygame.Rect(left, top, w, h)

        self.text_box = self.rendered_text.get_rect(center=self.rectangle.center)
        self.value = value

    def select(self, selection):
        self.selected = selection
        self.draw()

    def select_from_coord(self, coord):
        if self.rectangle.collidepoint(coord):
            self.select(True)
            return True
        return False

    def update_rectangle(self):
        self.rectangle.left = self.left
        self.rectangle.top = self.top
        self.text_box = self.rendered_text.get_rect(center=self.rectangle.center)

    def draw(self):
        color = self.selection_color if self.selected else self.background_color
        pygame.draw.rect(self.display, color, self.rectangle)
        self.display.blit(self.rendered_text, self.text_box)


class ButtonsGroup:
    def __init__(self, buttons_list=None, selected_index=0, space=10, left=0, top=0):
        if buttons_list is None:
            buttons_list = []
        self.buttons_list = buttons_list
        self.selected_index = selected_index
        self.buttons_list[self.selected_index].selected = True
        self.top = top
        self.left = left
        current_pos = self.left
        for button in self.buttons_list:
            button.top = self.top
            button.left = current_pos
            button.update_rectangle()
            current_pos += (space + button.w)

    def select_from_coord(self, coord):
        for idx, button in enumerate(self.buttons_list):
            if button.select_from_coord(coord):
                if idx != self.selected_index:
                    self.buttons_list[self.selected_index].select(False)
                self.selected_index = idx
                return True
        return False

    def draw(self):
        for button in self.buttons_list:
            button.draw()

    def get_value(self):
        return self.buttons_list[self.selected_index].value


def draw_text(display, text, y, x, font_size=32):
    font = pygame.font.SysFont('arial', font_size)
    text = font.render(text, True, (0, 0, 0))
    text_box = text.get_rect()
    text_box.top = y
    text_box.left = x
    display.blit(text, text_box)


def draw_options():
    display = pygame.display.set_mode(size=(325, 415))
    display.fill((180, 182, 177))
    draw_text(display, "Board size:", 20, 30, 20)
    board_size_button = ButtonsGroup(
        top=50,
        left=30,
        buttons_list=[
            Button(display=display, w=80, h=30, text="7x7", value=7),
            Button(display=display, w=80, h=30, text="9x9", value=9),
            Button(display=display, w=80, h=30, text="11x11", value=11),
        ],
        selected_index=0)

    draw_text(display, "Player color:", 90, 30, 20)
    color_button = ButtonsGroup(
        top=120,
        left=30,
        buttons_list=[
            Button(display=display, w=80, h=30, text="blue", value="blue"),
            Button(display=display, w=80, h=30, text="red", value="red")
        ],
        selected_index=0)

    draw_text(display, "Algorithm:", 160, 30, 20)
    alg_button = ButtonsGroup(
        top=190,
        left=30,
        buttons_list=[
            Button(display=display, w=80, h=35, text="minimax", value="minimax"),
            Button(display=display, w=80, h=35, text="alpha-beta", value="alpha-beta")
        ],
        selected_index=0)

    draw_text(display, "Difficulty:", 240, 30, 20)
    difficulty_button = ButtonsGroup(
        top=270,
        left=30,
        buttons_list=[
            Button(display=display, w=80, h=35, text="easy", value=2),
            Button(display=display, w=80, h=35, text="medium", value=3),
            Button(display=display, w=80, h=35, text="hard", value=4),
        ],
        selected_index=0)

    ok = Button(display=display, top=350, left=240, w=50, h=30, text="Ok", background_color=(155, 0, 55))
    board_size_button.draw()
    color_button.draw()
    alg_button.draw()
    difficulty_button.draw()
    ok.draw()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if not board_size_button.select_from_coord(pos):
                    if not color_button.select_from_coord(pos):
                        if not alg_button.select_from_coord(pos):
                            if not difficulty_button.select_from_coord(pos):
                                if ok.select_from_coord(pos):
                                    display.fill((0, 0, 0))
                                    pygame.display.update()
                                    # must draw initial state
                                    return board_size_button.get_value(), \
                                        color_button.get_value(), alg_button.get_value(), \
                                        difficulty_button.get_value()
        pygame.display.update()


def make_player_move(state, row1, col1, row2, col2):
    matrix = state.game.board.get_matrix()
    king, pieces, corners = state.game.board.get_pieces(Game.P_MIN)
    initial_piece = Piece(matrix[row1][col1], (row1, col1))
    moved_piece = None
    for piece in pieces:
        if piece.position == (row1, col1):
            if piece.position != king.position:
                moved_piece = Piece(piece.type, (row2, col2))
                break

    if moved_piece is None:
        if king.position == (row1, col1):
            if king.type == 'B' and matrix[row2][col2] == 'b':
                moved_piece = Piece('U', (row2, col2))
            elif king.type == 'B' and matrix[row2][col2] == '#':
                moved_piece = Piece('B', (row2, col2))
            elif king.type == 'R' and matrix[row2][col2] == 'r':
                moved_piece = Piece('U', (row2, col2))
            elif king.type == 'R' and matrix[row2][col2] == '#':
                moved_piece = Piece('R', (row2, col2))
            elif king.type == 'U' and matrix[row2][col2] == 'b':
                moved_piece = Piece('U', (row2, col2))
            elif king.type == 'U' and matrix[row2][col2] == '#':
                moved_piece = Piece('B', (row2, col2))
            elif king.type == 'V' and matrix[row2][col2] == 'r':
                moved_piece = Piece('V', (row2, col2))
            elif king.type == 'V' and matrix[row2][col2] == '#':
                moved_piece = Piece('R', (row2, col2))

    if moved_piece is None:
        return False, None

    possible_moves = state.game.possible_moves(state.current_player)

    for possible_move in possible_moves:
        if initial_piece.type == possible_move[0].type and initial_piece.position == possible_move[0].position:
            if possible_move[1].position == moved_piece.position and possible_move[1].type == moved_piece.type:
                return True, possible_move[2]
    return False, None


def init_game():
    pygame.init()
    pygame.display.set_caption("Four corners - Matei Dorian Nastase")


def print_winner(state):
    status = state.game.final_state()
    if status:
        print("\nThe winner is " + status)
        return True
    return False


def run_game():
    w = 52
    n, Game.P_MIN, algorithm_type, difficulty = draw_options()
    display = pygame.display.set_mode(size=(n * (w + 1) + 60, n * (w + 1) + 60))  # 60 is the padding around the board
    Game.get_start_state(display, size=n, cell_size=w)
    border = pygame.Rect(25, 25, n * (w + 1) + 9, n * (w + 1) + 9)
    pygame.draw.rect(display, (160, 89, 38), border, 5)
    pygame.display.update()
    current_game = Game(size=n)
    print(n, Game.P_MIN, algorithm_type, difficulty)

    Game.P_MAX = 'blue' if Game.P_MIN == 'red' else 'red'

    current_state = State(current_game, 'blue', difficulty)
    current_game.draw_grid()

    while True:
        if current_state.game.final_state():
            print("The winner is " + Game.opposite_player(current_state.current_player))
            pygame.quit()
            sys.exit()
        if current_state.current_player == Game.P_MIN:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for np in range(len(Game.cells)):
                        if Game.cells[np].collidepoint(pos):
                            row = np // Game.SIZE % Game.SIZE
                            col = np % Game.SIZE
                            shape_surf = pygame.Surface(pygame.Rect(Game.cells[np]).size, pygame.SRCALPHA)
                            pygame.draw.rect(shape_surf, (0, 78, 56, 60), shape_surf.get_rect())
                            display.blit(shape_surf, Game.cells[np])
                            pygame.display.update()
                            process_next_click = False
                            while not process_next_click:
                                for event2 in pygame.event.get():
                                    if event2.type == pygame.QUIT:
                                        pygame.quit()
                                        sys.exit()
                                    elif event2.type == pygame.MOUSEBUTTONDOWN:
                                        next_pos = pygame.mouse.get_pos()
                                        for np2 in range(len(Game.cells)):
                                            if Game.cells[np2].collidepoint(next_pos):
                                                next_row = np2 // Game.SIZE % Game.SIZE
                                                next_col = np2 % Game.SIZE
                                                status, new_board = make_player_move(current_state,
                                                                                     row, col, next_row, next_col)
                                                if status:
                                                    current_state.game.board = new_board
                                                    print("The board after the player's move")
                                                    print(str(current_state))
                                                    current_state.game.draw_grid()
                                                    print("\nPC's turn:")
                                                    if print_winner(current_state):
                                                        break
                                                    current_state.current_player = Game.opposite_player(
                                                        current_state.current_player)
                                                    process_next_click = True
                                                    break
                                                else:
                                                    current_state.game.draw_grid()
                                                    process_next_click = True
                                                    break

        else:
            start_t = int(round(time.time() * 1000))
            if algorithm_type == 'minimax':
                new_state = min_max(current_state)
            else:
                new_state = alpha_beta(-500, 500, current_state)
            current_state.game = new_state.next_state.game

            print("The board after the move of the PC\n" + str(current_state))

            final_t = int(round(time.time() * 1000))
            print("The pc \"thought\" about " + str(final_t - start_t) + " milliseconds.")

            current_state.game.draw_grid()
            pygame.display.update()
            if print_winner(current_state):
                break
            current_state.current_player = Game.opposite_player(current_state.current_player)
            print("\nYour turn:")


if __name__ == "__main__":
    init_game()
    run_game()
