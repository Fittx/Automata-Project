from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple
from .nfa import NFA, ALPHABET

# Order is intentional: it reproduces the project's D0..D34 numbering.
INPUT_ORDER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ;")

class DFA:
    """Reachable subset-construction DFA generated from the project's NFA."""

    def __init__(self, nfa: Optional[NFA] = None):
        self.nfa = nfa or NFA()
        self.states: List[FrozenSet[int]] = []
        self.state_index: Dict[FrozenSet[int], int] = {}
        self.transitions: Dict[Tuple[int, str], int] = {}
        self.start = 0
        self.final_states = set()
        self.dead_state: Optional[int] = None
        self._build()

    def _build(self):
        start_set = self.nfa.epsilon_closure({self.nfa.start})
        self.states = [start_set]
        self.state_index = {start_set: 0}
        queue = deque([start_set])

        while queue:
            subset = queue.popleft()
            src = self.state_index[subset]

            for ch in INPUT_ORDER:
                moved = self.nfa.move(set(subset), ch)
                if not moved:
                    continue

                target = self.nfa.epsilon_closure(moved)
                if target not in self.state_index:
                    self.state_index[target] = len(self.states)
                    self.states.append(target)
                    queue.append(target)

                self.transitions[(src, ch)] = self.state_index[target]

        self.final_states = {
            i for i, subset in enumerate(self.states)
            if set(subset) & self.nfa.final_states
        }

        # This language must produce exactly the 35 reachable DFA states
        # shown in the project's DFA specification.
        if len(self.states) != 35:
            raise RuntimeError(
                f"Subset construction produced {len(self.states)} states; expected 35."
            )
        # A single empty-subset trap state totalizes the DFA.  It is appended
        # after the documented D0..D34 subset-construction states.
        self.dead_state = len(self.states)
        self.states.append(frozenset())
        for state in range(self.dead_state):
            for ch in INPUT_ORDER:
                self.transitions.setdefault((state, ch), self.dead_state)
        for ch in INPUT_ORDER:
            self.transitions[(self.dead_state, ch)] = self.dead_state


    @property
    def state_count(self) -> int:
        return len(self.states)

    def transition(self, state: int, ch: str) -> Optional[int]:
        return self.transitions.get((state, ch))

    def accepts(self, text: str) -> bool:
        state = self.start
        for ch in text:
            nxt = self.transition(state, ch)
            state = nxt
        return state in self.final_states

    def trace(self, text: str) -> List[dict]:
        state = self.start
        trace = [{
            "step": 0,
            "input": "START",
            "state": f"D{state}",
            "state_id": state,
            "edge": None,
        }]

        for step, ch in enumerate(text, 1):
            nxt = self.transition(state, ch)
            state = nxt
            trace.append({
                "step": step,
                "input": "SPACE" if ch == " " else ch,
                "char": ch,
                "state": "qd" if state == self.dead_state else f"D{state}",
                "state_id": state,
                "from_state": trace[-1].get("state_id"),
                "edge": (trace[-1].get("state_id"), state),
            })

        return trace

    def grouped_edges(self) -> List[Tuple[int, int, str]]:
        """Group equivalent visible input labels between the same DFA nodes."""
        buckets: Dict[Tuple[int, int], List[str]] = {}
        for (src, ch), dst in self.transitions.items():
            buckets.setdefault((src, dst), []).append(ch)

        result = []
        for (src, dst), chars in buckets.items():
            # Missing transitions are visualized only when that input uses
            # one; rendering every trap arrow would obscure the full graph.
            if dst == self.dead_state and src != self.dead_state:
                continue
            result.append((src, dst, "Σ" if src == self.dead_state else self._compress_labels(chars)))
        return sorted(result, key=lambda x: (x[0], x[1], x[2]))

    @staticmethod
    def _compress_labels(chars: List[str]) -> str:
        s = set(chars)
        parts = []

        upper = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        lower = set("abcdefghijklmnopqrstuvwxyz")
        digits = set("0123456789")

        if lower and lower <= s:
            parts.append("[a-z]")
            s -= lower
        if digits and digits <= s:
            parts.append("[0-9]")
            s -= digits

        # Keep uppercase letters and punctuation readable.
        for ch in sorted(s, key=lambda x: (x == " ", x)):
            parts.append("SPACE" if ch == " " else ch)

        return ", ".join(parts)

    def nfa_set_label(self, state: int) -> str:
        return "{" + ", ".join(f"q{x}" for x in sorted(self.states[state])) + "}"
