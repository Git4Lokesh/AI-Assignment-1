# You need to complete this function which should return action,

from game.go import Board, opponent_color, neighbors, cal_liberty
import random


class Agent1:
    """A class to generate a strategic action for a Go board."""

    def __init__(self, color):
        self.color = color.upper()  # Ensure uppercase

    def get_action(self, board: Board):
        """
        Returns a strategic legal action from the board.

        :param board: The current Go board state.
        :return: A strategic legal action (tuple) or None if no actions are available.
        """
        actions = board.get_legal_actions()
        if not actions:
            return None

        # Strategy 1: Prioritize winning moves (capturing opponent groups)
        winning_moves = self._get_winning_moves(board, actions)
        if winning_moves:
            return random.choice(winning_moves)

        # Strategy 2: Defend endangered self-groups
        defense_moves = self._get_defense_moves(board, actions)
        if defense_moves:
            return random.choice(defense_moves)

        # Strategy 3: Attack opponent's weak groups (with few liberties)
        attack_moves = self._get_attack_moves(board, actions)
        if attack_moves:
            return random.choice(attack_moves)

        # Strategy 4: Choose moves with maximum liberties (safer positions)
        best_moves = self._get_moves_with_max_liberties(board, actions)
        if best_moves:
            return random.choice(best_moves)

        # Fallback to random move
        return random.choice(actions)

    def _get_winning_moves(self, board, actions):
        """Find moves that capture opponent groups (liberties == 0 after move)."""
        winning_moves = []
        opponent = opponent_color(self.color)
        
        for action in actions:
            # Check if this action removes liberty from opponent groups
            opponent_groups = board.libertydict.get_groups(opponent, action)
            for group in opponent_groups:
                if len(group.liberties) == 1:  # This is the last liberty
                    winning_moves.append(action)
                    break
        
        return winning_moves

    def _get_defense_moves(self, board, actions):
        """Find moves that save endangered self-groups."""
        defense_moves = []
        
        for action in actions:
            # Check if this action adds liberties to endangered self-groups
            self_groups = board.libertydict.get_groups(self.color, action)
            for group in self_groups:
                if len(group.liberties) <= 2:  # Group is endangered or close to it
                    defense_moves.append(action)
                    break
        
        return defense_moves

    def _get_attack_moves(self, board, actions):
        """Find moves that threaten opponent groups with few liberties."""
        attack_moves = []
        opponent = opponent_color(self.color)
        
        for action in actions:
            # Check if this action reduces liberties of opponent groups
            opponent_groups = board.libertydict.get_groups(opponent, action)
            for group in opponent_groups:
                if len(group.liberties) <= 3:  # Target weak opponent groups
                    attack_moves.append(action)
                    break
        
        return attack_moves

    def _get_moves_with_max_liberties(self, board, actions):
        """Find moves that result in maximum liberties after placement."""
        max_liberties = -1
        best_moves = []
        
        for action in actions:
            # Calculate liberties for this action
            liberties = len(cal_liberty(action, board))
            
            # Check if connected to friendly groups with good liberties
            for neighbor in neighbors(action):
                self_groups = board.stonedict.get_groups(self.color, neighbor)
                if self_groups:
                    liberties += len(self_groups[0].liberties) - 1
                    break
            
            if liberties > max_liberties:
                max_liberties = liberties
                best_moves = [action]
            elif liberties == max_liberties:
                best_moves.append(action)
        
        return best_moves
