from game.go import Board, opponent_color, neighbors, cal_liberty
import random


class Agent1:

    def __init__(self, color):
        self.color = color.upper()
        self.max_depth = 3
        self.transposition_table = {}

    def get_action(self, board: Board):
        actions = board.get_legal_actions()
        if not actions:
            return None
        
        if len(self.transposition_table) > 10000:
            self.transposition_table.clear()
        
        best_action = None
        for depth in range(1, self.max_depth + 1):
            action, _ = self.minimax(board, depth, float('-inf'), float('inf'), True)
            if action is not None:
                best_action = action
        
        return best_action if best_action else random.choice(actions)

    def get_board_hash(self, board):
        return hash((
            tuple(sorted(tuple(g.points) for g in board.groups[self.color])),
            tuple(sorted(tuple(g.points) for g in board.groups[opponent_color(self.color)])),
            board.next
        ))

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        board_hash = self.get_board_hash(board)
        if board_hash in self.transposition_table:
            cached_depth, cached_value, cached_action = self.transposition_table[board_hash]
            if cached_depth >= depth:
                return cached_action, cached_value
        
        if depth == 0 or board.winner:
            eval_score = self.evaluate(board)
            return None, eval_score
        
        actions = board.get_legal_actions()
        if not actions:
            return None, self.evaluate(board)
        
        ordered_actions = self._order_moves_advanced(board, actions)
        
        if maximizing_player:
            max_eval = float('-inf')
            best_action = None
            
            for action in ordered_actions:
                successor = board.generate_successor_state(action)
                _, eval_score = self.minimax(successor, depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            self.transposition_table[board_hash] = (depth, max_eval, best_action)
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
                    break
            
            self.transposition_table[board_hash] = (depth, min_eval, best_action)
            return best_action, min_eval

    def evaluate(self, board):
        if board.winner == self.color:
            return 100000
        elif board.winner == opponent_color(self.color):
            return -100000
        
        score = 0
        opponent = opponent_color(self.color)
        
        self_score = self._evaluate_player(board, self.color)
        opponent_score = self._evaluate_player(board, opponent)
        score = self_score - opponent_score
        
        score += self._evaluate_tactical_threats(board)
        score += self._evaluate_influence_and_territory(board)
        score += self._evaluate_cutting_points(board)
        score += self._evaluate_edge_positions(board)
        
        return score
    
    def _evaluate_player(self, board, color):
        score = 0
        
        for group in board.groups[color]:
            group_size = len(group.points)
            liberties = len(group.liberties)
            
            if liberties == 1:
                score -= 250 * group_size 
            elif liberties == 2:
                score -= 80 * (group_size ** 0.5)
            elif liberties == 3:
                score += 30
            elif liberties == 4:
                score += 70
            elif liberties >= 5:
                score += 110 + (liberties - 5) * 5
            
            if group_size <= 3:
                score += group_size * 18
            elif group_size <= 6:
                score += 54 + (group_size - 3) * 12
            else:
                score += 90 + (group_size - 6) * 8
            
            connectivity = self._evaluate_group_connectivity(board, group)
            score += connectivity * 10
            
            eye_potential = self._evaluate_eye_potential(board, group, color)
            score += eye_potential * 20
            
            if group_size == 1:
                isolated = True
                for neighbor_pos in neighbors(list(group.points)[0]):
                    if board.stonedict.get_groups(color, neighbor_pos):
                        isolated = False
                        break
                if isolated:
                    score -= 5
        
        return score
    
    def _evaluate_group_connectivity(self, board, group):
        if len(group.points) <= 1:
            return 0
        
        connectivity = 0
        for point in group.points:
            for other_point in group.points:
                if point != other_point and other_point in neighbors(point):
                    connectivity += 1
        
        return connectivity // 2
    
    def _evaluate_eye_potential(self, board, group, color):
        if len(group.liberties) < 2:
            return 0
        
        eye_potential = 0
        liberties_list = list(group.liberties)
        
        for i, lib1 in enumerate(liberties_list):
            for lib2 in liberties_list[i+1:]:
                if lib2 in neighbors(lib1):
                    lib1_neighbors = neighbors(lib1)
                    lib2_neighbors = neighbors(lib2)
                    
                    friendly_count = sum(
                        1 for n in set(lib1_neighbors + lib2_neighbors)
                        if board.stonedict.get_groups(color, n)
                    )
                    
                    if friendly_count >= 5:
                        eye_potential += 3
                    elif friendly_count >= 4:
                        eye_potential += 2
                    elif friendly_count >= 3:
                        eye_potential += 1
        
        return eye_potential
    
    def _evaluate_tactical_threats(self, board):
        score = 0
        opponent = opponent_color(self.color)
        
        for group in board.groups[opponent]:
            liberties = len(group.liberties)
            group_size = len(group.points)
            
            if liberties == 1:
                score += 350 + group_size * 25
            elif liberties == 2:
                score += 100 + group_size * 8
            elif liberties == 3:
                score += 30
        
        for group in board.groups[self.color]:
            liberties = len(group.liberties)
            group_size = len(group.points)
            
            if liberties == 1:
                score -= 350 + group_size * 25
            elif liberties == 2:
                score -= 100 + group_size * 8
            elif liberties == 3:
                score -= 30
        
        return score
    
    def _evaluate_influence_and_territory(self, board):
        score = 0
        opponent = opponent_color(self.color)
        
        self_influence = set()
        opponent_influence = set()
        
        for group in board.groups[self.color]:
            self_influence.update(group.liberties)
        
        for group in board.groups[opponent]:
            opponent_influence.update(group.liberties)
        
        uncontested_self = self_influence - opponent_influence
        uncontested_opponent = opponent_influence - self_influence
        
        score += len(uncontested_self) * 6
        score -= len(uncontested_opponent) * 6
        score += len(self_influence & opponent_influence) * 2
        
        board_size = 20
        center = (board_size // 2, board_size // 2)
        
        for point in self_influence:
            distance = abs(point[0] - center[0]) + abs(point[1] - center[1])
            if distance <= 3:
                score += 10
            elif distance <= 6:
                score += 5
            elif distance <= 9:
                score += 2
        
        return score
    
    def _evaluate_cutting_points(self, board):
        score = 0
        opponent = opponent_color(self.color)
        
        for group in board.groups[opponent]:
            for liberty in group.liberties:
                adjacent_opponent_groups = set()
                for neighbor_pos in neighbors(liberty):
                    groups = board.stonedict.get_groups(opponent, neighbor_pos)
                    if groups:
                        adjacent_opponent_groups.update(groups)
                
                if len(adjacent_opponent_groups) >= 2:
                    score += 40
        
        return score
    
    def _evaluate_edge_positions(self, board):
        score = 0
        board_size = 20
        
        corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
        
        for corner in corners:
            self_near_corner = any(
                board.stonedict.get_groups(self.color, (corner[0] + dx, corner[1] + dy))
                for dx in range(-2, 3) for dy in range(-2, 3)
                if 0 <= corner[0] + dx < board_size and 0 <= corner[1] + dy < board_size
            )
            
            if self_near_corner:
                score += 15
        
        return score

    def _order_moves_advanced(self, board, actions):
        move_scores = []
        opponent = opponent_color(self.color)
        
        for action in actions:
            score = 0
            
            opponent_groups = board.libertydict.get_groups(opponent, action)
            for group in opponent_groups:
                if len(group.liberties) == 1:
                    score += 1000 + len(group.points) * 50
                elif len(group.liberties) == 2:
                    score += 200
                elif len(group.liberties) == 3:
                    score += 50
            
            self_groups = board.libertydict.get_groups(self.color, action)
            for group in self_groups:
                if len(group.liberties) == 1:
                    score += 800 + len(group.points) * 40
                elif len(group.liberties) == 2:
                    score += 150
            
            center = (10, 10)
            distance = abs(action[0] - center[0]) + abs(action[1] - center[1])
            score += max(0, 20 - distance)
            
            move_scores.append((score, action))
        
        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [action for _, action in move_scores]