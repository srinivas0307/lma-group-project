"""Orchestrator states and the only allowed transitions (README §4.6). Every run records its path; an illegal
transition raises, so a bug in the agent loop can't silently skip the safety check or the verifier."""

TRANSITIONS = {
    "PREPARE": {"PARSE", "GENERATE"},
    "PARSE": {"ESCALATED", "GENERATE"},
    "GENERATE": {"SAFETY_CHECK"},
    "SAFETY_CHECK": {"EXECUTE", "FAILED"},
    "EXECUTE": {"VERIFY", "FAILED"},
    "VERIFY": {"ACCEPTED", "ESCALATED", "FAILED"},
    "FAILED": {"REPAIR", "REJECTED"},
    "REPAIR": {"REPAIR", "SAFETY_CHECK", "ESCALATED", "REJECTED"},   # REJECTED: token budget / step cap
    "ACCEPTED": set(), "REJECTED": set(), "ESCALATED": set(),
}
FINAL = {"ACCEPTED": "ACCEPT", "REJECTED": "REJECT", "ESCALATED": "ESCALATE"}


class IllegalTransition(RuntimeError):
    pass


class StateMachine:
    def __init__(self, start="PREPARE"):
        self.state = start
        self.path = [start]

    def go(self, new):
        if new not in TRANSITIONS[self.state]:
            raise IllegalTransition(f"{self.state} -> {new} is not allowed")
        self.state = new
        self.path.append(new)
        return new

    @property
    def done(self):
        return self.state in FINAL

    @property
    def outcome(self):
        return FINAL.get(self.state)
