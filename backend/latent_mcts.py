import math
import asyncio
import torch
import torch.nn as nn

class Node:
    def __init__(self, prior: float = 0.0, reward: float = 0.0, hidden_state: torch.Tensor = None):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.reward = reward
        self.hidden_state = hidden_state
        self.children = {}  # dict mapping action (int) -> Node

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class LatentMCTS:
    def __init__(self, model: nn.Module, c_puct: float = 1.25):
        self.model = model
        self.c_puct = c_puct

    def puct_score(self, parent_visit_count: int, child: Node) -> float:
        q_value = child.value()
        # PUCT formula: Q(s, a) + c_puct * P(s, a) * sqrt(N_parent) / (1 + N_child)
        u_value = self.c_puct * child.prior * (math.sqrt(parent_visit_count) / (1 + child.visit_count))
        return q_value + u_value

    async def search(self, initial_state_tensor: torch.Tensor, legal_actions: list, num_simulations: int = 50, websocket=None, stop_event=None):
        self.model.eval()
        device = initial_state_tensor.device
        board_size = initial_state_tensor.shape[-1]
        action_space_size = board_size * board_size

        # 1. Initial Inference from Representation Network
        with torch.no_grad():
            init_val, init_rw, init_policy_logits, init_latent_state = self.model.initial_inference(initial_state_tensor)
            priors = torch.softmax(init_policy_logits[0], dim=-1)

        # Mask illegal actions and re-normalize priors
        legal_priors = {}
        total_p_sum = 0.0
        for act in legal_actions:
            p_val = priors[act].item() if act < len(priors) else 0.0
            legal_priors[act] = p_val
            total_p_sum += p_val

        # Create Root Node
        root = Node(prior=1.0, reward=0.0, hidden_state=init_latent_state)

        for act in legal_actions:
            norm_prior = (legal_priors[act] / total_p_sum) if total_p_sum > 0 else (1.0 / len(legal_actions))
            root.children[act] = Node(prior=norm_prior, reward=0.0, hidden_state=None)

        # 2. Main MCTS Simulation Loop
        for sim in range(num_simulations):
            if stop_event and stop_event.is_set():
                break

            node = root
            path = []

            # --- SELECT ---
            while len(node.children) > 0:
                # Find unexpanded child or pick best via PUCT
                unexpanded = [act for act, child in node.children.items() if child.hidden_state is None]
                if unexpanded:
                    # Pick first unexpanded child for evaluation
                    action = unexpanded[0]
                    path.append((node, action))
                    node = node.children[action]
                    break
                else:
                    # All children expanded, select child with max PUCT
                    best_score = -float('inf')
                    best_act = None
                    for act, child in node.children.items():
                        score = self.puct_score(node.visit_count, child)
                        if score > best_score:
                            best_score = score
                            best_act = act

                    if best_act is None:
                        break

                    path.append((node, best_act))
                    node = node.children[best_act]

            # --- EXPAND & EVALUATE ---
            leaf_value = 0.0
            if node.hidden_state is None and len(path) > 0:
                parent_node, chosen_act = path[-1]
                with torch.no_grad():
                    action_tensor = torch.tensor([chosen_act], device=device, dtype=torch.long)
                    rec_val, rec_rw, rec_pol_logits, next_latent = self.model.recurrent_inference(
                        parent_node.hidden_state, action_tensor
                    )
                    rec_priors = torch.softmax(rec_pol_logits[0], dim=-1)

                node.hidden_state = next_latent
                node.reward = rec_rw.item()
                leaf_value = rec_val.item()

                # Expand children for legal actions
                for act in legal_actions:
                    p_val = rec_priors[act].item() if act < len(rec_priors) else 0.0
                    node.children[act] = Node(prior=p_val, reward=0.0, hidden_state=None)
            else:
                leaf_value = init_val.item()

            # --- BACKPROPAGATE ---
            # CRITICAL: Invert value at each step up the tree for alternating zero-sum games
            value = leaf_value
            for parent, act in reversed(path):
                child_node = parent.children[act]
                child_node.visit_count += 1
                child_node.value_sum += value
                value = -value  # Invert value for alternating player perspective

            root.visit_count += 1

            # --- WEBSOCKET STREAMING ---
            if websocket and (sim + 1) % 10 == 0:
                await asyncio.sleep(0.001)
                visits_list = [
                    {"move": act, "visits": child.visit_count}
                    for act, child in root.children.items()
                ]
                try:
                    await websocket.send_json({
                        "type": "heatmap_update",
                        "total_nodes": root.visit_count,
                        "visits": visits_list
                    })
                except Exception:
                    pass

        return root
