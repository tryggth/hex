import math
import random
import numpy as np

class Node:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.untried_actions = self.state.legal_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state.winner != 0

class ClassicMCTS:
    def __init__(self, c_puct=1.414):
        self.c_puct = c_puct

    def search(self, env_state, num_simulations, temperature=0.0):
        root = Node(env_state.clone())

        for _ in range(num_simulations):
            node = root

            # 1. Select
            while node.is_fully_expanded() and not node.is_terminal():
                best_score = -float('inf')
                best_action = None
                best_child = None
                
                for act, child in node.children.items():
                    if child.visit_count == 0:
                        score = float('inf')
                    else:
                        score = (child.value_sum / child.visit_count) + \
                                self.c_puct * math.sqrt(math.log(node.visit_count) / child.visit_count)
                    
                    if score > best_score:
                        best_score = score
                        best_action = act
                        best_child = child
                
                node = best_child

            # 2. Expand
            if not node.is_terminal():
                action_idx = random.randint(0, len(node.untried_actions) - 1)
                action = node.untried_actions.pop(action_idx)
                
                new_state = node.state.clone()
                new_state.step(action)
                
                child_node = Node(new_state, parent=node, action_taken=action)
                node.children[action] = child_node
                node = child_node

            # 3. Simulate
            sim_state = node.state.clone()
            while sim_state.winner == 0:
                legal_moves = sim_state.legal_actions()
                if not legal_moves:
                    break
                act = random.choice(legal_moves)
                sim_state.step(act)
                
            player_who_just_moved = 3 - node.state.current_player
            if sim_state.winner == player_who_just_moved:
                reward = 1.0
            elif sim_state.winner != 0:
                reward = -1.0
            else:
                reward = 0.0

            # 4. Backpropagate
            curr = node
            curr_reward = reward
            while curr is not None:
                curr.visit_count += 1
                curr.value_sum += curr_reward
                curr_reward = -curr_reward
                curr = curr.parent

        target_policy = np.zeros(env_state.num_cells, dtype=np.float32)
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits > 0:
            for act, child in root.children.items():
                target_policy[act] = child.visit_count / total_visits
        else:
            legal = env_state.legal_actions()
            for act in legal:
                target_policy[act] = 1.0 / len(legal)

        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)

        if np.sum(visits) > 0:
            if temperature > 0.0:
                visits_norm = visits / np.max(visits)
                visits_pow = visits_norm ** (1.0 / temperature)
                probs = visits_pow / np.sum(visits_pow)
                probs = probs / np.sum(probs)
                best_action = np.random.choice(actions, p=probs)
            else:
                best_action = max(actions, key=lambda a: root.children[a].visit_count)
        else:
            legal = env_state.legal_actions()
            best_action = random.choice(legal) if legal else None
                
        return best_action, target_policy
