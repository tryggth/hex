import numpy as np

HEX_DIRS = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]]

class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i: int) -> int:
        root = i
        while root != self.parent[root]:
            root = self.parent[root]
        curr = i
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, i: int, j: int):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

class HexEnv:
    def __init__(self, board_size: int = 7):
        self.board_size = board_size
        self.num_cells = board_size * board_size

        # Sentinels
        self.red_top = self.num_cells
        self.red_bottom = self.num_cells + 1
        self.blue_left = self.num_cells + 2
        self.blue_right = self.num_cells + 3

        self.reset()

    def reset(self):
        self.board = np.zeros(self.num_cells, dtype=np.uint8)
        self.uf = UnionFind(self.num_cells + 4)
        self.current_player = 1  # 1: Red (Top-to-Bottom), 2: Blue (Left-to-Right)
        self.winner = 0
        self.moves_made = 0
        return self.get_observation()

    def is_valid(self, r: int, c: int) -> bool:
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def coord_to_idx(self, r: int, c: int) -> int:
        return r * self.board_size + c

    def legal_actions(self) -> list:
        return [i for i in range(self.num_cells) if self.board[i] == 0]

    def step(self, action: int):
        if self.board[action] != 0 or self.winner != 0:
            raise ValueError(f"Invalid move index {action} on board state")

        r = action // self.board_size
        c = action % self.board_size
        player = self.current_player

        self.board[action] = player
        self.moves_made += 1

        # Connect to adjacent friendly stones
        for dr, dc in HEX_DIRS:
            nr, nc = r + dr, c + dc
            if self.is_valid(nr, nc):
                nidx = self.coord_to_idx(nr, nc)
                if self.board[nidx] == player:
                    self.uf.union(action, nidx)

        # Connect border cells to sentinels
        if player == 1:  # Red (Top-to-Bottom)
            if r == 0:
                self.uf.union(action, self.red_top)
            if r == self.board_size - 1:
                self.uf.union(action, self.red_bottom)
        else:  # Blue (Left-to-Right)
            if c == 0:
                self.uf.union(action, self.blue_left)
            if c == self.board_size - 1:
                self.uf.union(action, self.blue_right)

        # Check win conditions
        reward = 0.0
        done = False

        if player == 1 and self.uf.find(self.red_top) == self.uf.find(self.red_bottom):
            self.winner = 1
            reward = 1.0
            done = True
        elif player == 2 and self.uf.find(self.blue_left) == self.uf.find(self.blue_right):
            self.winner = 2
            reward = 1.0
            done = True
        elif self.moves_made == self.num_cells:
            done = True

        # Swap player turn
        self.current_player = 2 if self.current_player == 1 else 1

        return self.get_observation(), reward, done

    def get_observation(self) -> np.ndarray:
        obs = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        grid = self.board.reshape((self.board_size, self.board_size))

        # Channel 0: Player 1 (Red) stones
        obs[0] = (grid == 1).astype(np.float32)
        # Channel 1: Player 2 (Blue) stones
        obs[1] = (grid == 2).astype(np.float32)
        # Channel 2: Current player turn (1.0 for P1, 0.0 for P2)
        obs[2] = 1.0 if self.current_player == 1 else 0.0

        return obs
