# You need to complete this function which should return action,

from game.go import Board, opponent_color, neighbors, cal_liberty
import random


class Agent1:
    """A class to generate a strategic action for a Go board using Minimax with Alpha-Beta Pruning."""

    def __init__(self, color):
        self.color = color.upper()  # Ensure uppercase
        self.max_depth = 5  # Search depth (2-3 moves ahead)

    def get_action(self, board: Board):
        """
        Returns the best legal action using minimax with alpha-beta pruning.

        :param board: The current Go board state.
        :return: A strategic legal action (tuple) or None if no actions are available.
        """
        actions = board.get_legal_actions()
        if not actions:
            return None
        
        # Use minimax with alpha-beta pruning
        best_action, _ = self.minimax(board, self.max_depth, float('-inf'), float('inf'), True)
        
        # Fallback to random if minimax returns None
        if best_action is None:
            return random.choice(actions)
        
        return best_action

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning.
        
        :param board: Current board state
        :param depth: Remaining search depth
        :param alpha: Alpha value for pruning
        :param beta: Beta value for pruning
        :param maximizing_player: True if maximizing, False if minimizing
        :return: (best_action, best_value)
        """
        # Base case: depth limit reached or game over
        if depth == 0 or board.winner:
            return None, self.evaluate(board)
        
        actions = board.get_legal_actions()
        if not actions:
            return None, self.evaluate(board)
        
        # Order moves for better pruning (captures and threats first)
        ordered_actions = self._order_moves(board, actions)
        
        if maximizing_player:
            max_eval = float('-inf')
            best_action = None
            
            for action in ordered_actions:
                # Generate successor state
                successor = board.generate_successor_state(action)
                _, eval_score = self.minimax(successor, depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff (pruning)
            
            return best_action, max_eval
        else:
            min_eval = float('inf')
            best_action = None
            
            for action in ordered_actions:
                successor = board.generate_successor_state(action)
                _, eval_score = self.minimax(successor, depth - 1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff (pruning)
            
            return best_action, min_eval

    def evaluate(self, board):
        """
        Master-level heuristic evaluation function based on Go principles.
        Positive scores favor self, negative scores favor opponent.
        
        :param board: Board state to evaluate
        :return: Numerical score
        """
        # Terminal states
        if board.winner == self.color:
            return 100000  # Win
        elif board.winner == opponent_color(self.color):
            return -100000  # Loss
        
        score = 0
        opponent = opponent_color(self.color)
        
        # Evaluate both players
        self_score = self._evaluate_player(board, self.color)
        opponent_score = self._evaluate_player(board, opponent)
        
        score = self_score - opponent_score
        
        # Strategic modifiers
        score += self._evaluate_tactical_threats(board)
        score += self._evaluate_influence_and_territory(board)
        
        return score
    
    def _evaluate_player(self, board, color):
        """Evaluate position strength for a specific player."""
        score = 0
        
        for group in board.groups[color]:
            group_size = len(group.points)
            liberties = len(group.liberties)
            
            # 1. Liberty valuation (non-linear - critical in Go)
            # Based on Go principle: "Life and death" depends on liberties
            if liberties == 1:
                score -= 200  # Critical danger (atari)
            elif liberties == 2:
                score -= 50   # Significant danger
            elif liberties == 3:
                score += 20   # Relatively safe
            elif liberties == 4:
                score += 60   # Safe
            elif liberties >= 5:
                score += 100  # Very safe, excellent shape
            
            # 2. Group size with diminishing returns
            # Large groups are valuable but efficiency matters
            if group_size == 1:
                score += 10   # Single stone
            elif group_size == 2:
                score += 25   # Small group
            elif group_size <= 4:
                score += group_size * 15  # Medium group
            else:
                score += 60 + (group_size - 4) * 10  # Large group (diminishing returns)
            
            # 3. Shape quality (connectivity bonus)
            # Well-connected groups are stronger
            connectivity = self._evaluate_group_connectivity(board, group)
            score += connectivity * 8
            
            # 4. Eye potential (two liberties near each other = potential eye)
            eye_potential = self._evaluate_eye_potential(board, group, color)
            score += eye_potential * 15
        
        return score
    
    def _evaluate_group_connectivity(self, board, group):
        """
        Evaluate how well-connected a group's stones are.
        Better shape = more liberties shared between stones.
        """
        if len(group.points) <= 1:
            return 0
        
        connectivity = 0
        # Check how many stones share liberties (good shape indicator)
        for point in group.points:
            point_liberties = cal_liberty(point, board)
            for other_point in group.points:
                if point != other_point and other_point in neighbors(point):
                    connectivity += 1
        
        return connectivity // 2  # Divide by 2 since we count each connection twice
    
    def _evaluate_eye_potential(self, board, group, color):
        """
        Evaluate potential for creating eyes (enclosed liberties).
        Eyes are critical for group survival in Go.
        """
        if len(group.liberties) < 2:
            return 0
        
        eye_potential = 0
        liberties_list = list(group.liberties)
        
        # Check for liberties that are close together (potential eyes)
        for i, lib1 in enumerate(liberties_list):
            for lib2 in liberties_list[i+1:]:
                # If two liberties are neighbors, they form eye potential
                if lib2 in neighbors(lib1):
                    # Check if they're surrounded by friendly stones
                    lib1_neighbors = neighbors(lib1)
                    lib2_neighbors = neighbors(lib2)
                    
                    friendly_count = 0
                    for n in lib1_neighbors + lib2_neighbors:
                        if board.stonedict.get_groups(color, n):
                            friendly_count += 1
                    
                    if friendly_count >= 4:  # Strong eye potential
                        eye_potential += 2
                    elif friendly_count >= 3:
                        eye_potential += 1
        
        return eye_potential
    
    def _evaluate_tactical_threats(self, board):
        """
        Evaluate immediate tactical situations (atari, ladder, snapback).
        """
        score = 0
        opponent = opponent_color(self.color)
        
        # Check for immediate capture threats (atari)
        for group in board.groups[opponent]:
            if len(group.liberties) == 1:
                # Opponent in atari - huge advantage
                score += 300 + len(group.points) * 20  # More points = bigger capture
        
        for group in board.groups[self.color]:
            if len(group.liberties) == 1:
                # We're in atari - huge disadvantage
                score -= 300 + len(group.points) * 20
        
        # Two-liberty threats (opponent's perspective)
        opponent_weak_groups = 0
        self_weak_groups = 0
        
        for group in board.groups[opponent]:
            if len(group.liberties) == 2:
                opponent_weak_groups += 1
                score += 80  # Can potentially capture
        
        for group in board.groups[self.color]:
            if len(group.liberties) == 2:
                self_weak_groups += 1
                score -= 80  # Vulnerable to attack
        
        return score
    
    def _evaluate_influence_and_territory(self, board):
        """
        Evaluate influence/territory control using liberty dominance.
        """
        score = 0
        opponent = opponent_color(self.color)
        
        # Count total liberties controlled (influence metric)
        self_influence = set()
        opponent_influence = set()
        
        for group in board.groups[self.color]:
            self_influence.update(group.liberties)
        
        for group in board.groups[opponent]:
            opponent_influence.update(group.liberties)
        
        # Contested vs uncontested influence
        uncontested_self = self_influence - opponent_influence
        uncontested_opponent = opponent_influence - self_influence
        contested = self_influence & opponent_influence
        
        # Uncontested influence is more valuable
        score += len(uncontested_self) * 5
        score -= len(uncontested_opponent) * 5
        score += len(contested) * 1  # Contested areas have some value
        
        # Center control bonus (positions near center are more influential)
        center = (10, 10)
        for point in self_influence:
            distance = abs(point[0] - center[0]) + abs(point[1] - center[1])
            if distance <= 3:
                score += 8  # Central influence is valuable
            elif distance <= 6:
                score += 3
        
        return score

    def _order_moves(self, board, actions):
        """
        Order moves for better alpha-beta pruning efficiency.
        Prioritize: captures > threats > defense > other moves
        
        :param board: Current board state
        :param actions: List of legal actions
        :return: Ordered list of actions
        """
        capture_moves = []
        threat_moves = []
        defense_moves = []
        other_moves = []
        
        opponent = opponent_color(self.color)
        
        for action in actions:
            is_capture = False
            is_threat = False
            is_defense = False
            
            # Check if this is a capture move
            opponent_groups = board.libertydict.get_groups(opponent, action)
            for group in opponent_groups:
                if len(group.liberties) == 1:
                    is_capture = True
                    break
                elif len(group.liberties) <= 3:
                    is_threat = True
            
            # Check if this is a defense move
            if not is_capture:
                self_groups = board.libertydict.get_groups(self.color, action)
                for group in self_groups:
                    if len(group.liberties) <= 2:
                        is_defense = True
                        break
            
            # Categorize the move
            if is_capture:
                capture_moves.append(action)
            elif is_defense:
                defense_moves.append(action)
            elif is_threat:
                threat_moves.append(action)
            else:
                other_moves.append(action)
        
        # Return ordered list: captures first, then defense, then threats, then others
        return capture_moves + defense_moves + threat_moves + other_moves
