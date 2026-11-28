"""Task-local progress evidence for escalation, separate from provider errors."""


class StallTracker:
    def __init__(self):
        self.previous = None
        self.recall_streak = 0

    def observe(self, record):
        if record.passed or record.error:
            self.previous = None
            self.recall_streak = 0
            return False
        signature = tuple(sorted(record.gates.items()))
        repeated_failure = signature == self.previous
        explicit_stall = record.directive in {"RECALL_SINK", "SHIFT_COORDINATE"}
        self.previous = signature
        self.recall_streak = self.recall_streak + 1 if repeated_failure or explicit_stall else 0
        return self.recall_streak >= 1
