"""Deterministic, real multi-file Python repair fixtures.

Each fixture writes a tiny repository with pytest files, then exposes two or more
full-file candidate patches. The first candidate is deliberately incomplete; the
golden candidate fixes both sides of a cross-module bug. Needle-moving test files
and conftest are never in the editable allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

from .contracts import FileEdit, is_safe_relative, validated_edits

__all__ = ["BugFixture", "PatchCandidate", "builtin_fixtures", "tangled_fixtures"]


def _py(text: str) -> str:
    return dedent(text).lstrip()


def _fixture_files(
    *, package: str, sources: dict[str, str], test_name: str, test_source: str
) -> tuple[tuple[str, str], ...]:
    conftest = _py(
        """
        from __future__ import annotations

        import sys
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent
        sys.path.insert(0, str(ROOT))
        """
    )
    files: list[tuple[str, str]] = [
        ("conftest.py", conftest),
        (f"{package}/__init__.py", f"# {package} package\n"),
    ]
    files.extend((path, _py(source)) for path, source in sources.items())
    files.append((f"tests/{test_name}", _py(test_source)))
    return tuple(files)


@dataclass(frozen=True, slots=True)
class PatchCandidate:
    name: str
    edits: tuple[FileEdit, ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "edits": [edit.as_dict() for edit in self.edits],
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class BugFixture:
    id: str
    task: str
    files: tuple[tuple[str, str], ...]
    candidates: tuple[PatchCandidate, ...]
    probe_hint: str
    fix_index: int = 1

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError(f"fixture {self.id!r} needs at least one candidate")
        if not 0 <= self.fix_index < len(self.candidates):
            raise ValueError(f"fixture {self.id!r} has an invalid fix_index")
        allowed = self.editable_paths
        for path, _ in self.files:
            if not is_safe_relative(path):
                raise ValueError(f"fixture {self.id!r} contains unsafe path {path!r}")
        for candidate in self.candidates:
            validated_edits(candidate.edits, allowed)

    @property
    def file_map(self) -> dict[str, str]:
        return dict(self.files)

    @property
    def editable_paths(self) -> tuple[str, ...]:
        return tuple(
            path for path, _ in self.files if path != "conftest.py" and not path.startswith("tests/")
        )

    @property
    def test_targets(self) -> tuple[str, ...]:
        return tuple(
            path
            for path, _ in self.files
            if path.startswith("tests/") and path.endswith(".py") and Path(path).name.startswith("test_")
        )

    @property
    def fix_candidate(self) -> PatchCandidate:
        return self.candidates[self.fix_index]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "files": [path for path, _ in self.files],
            "editable_paths": list(self.editable_paths),
            "test_targets": list(self.test_targets),
            "fix_candidate": self.fix_candidate.name,
            "probe_hint": self.probe_hint,
        }


_CART_LEDGER_BUG = _py(
    """
    def subtotal(items):
        total = 1
        for item in items:
            total += item["price"] * item["qty"]
        return total


    def discount(amount, coupon):
        return amount * (1.0 - coupon)
    """
)

_CART_LEDGER_FIX = _py(
    """
    def subtotal(items):
        total = 0
        for item in items:
            total += item["price"] * item["qty"]
        return total


    def discount(amount, coupon):
        return amount * (1.0 - coupon)
    """
)

_CART_BUG = _py(
    """
    from .ledger import discount, subtotal


    def total(items, *, coupon=0.0):
        base = subtotal(items)
        return discount(base, coupon) + 1
    """
)

_CART_CART_ONLY_FIX = _py(
    """
    from .ledger import discount, subtotal


    def total(items, *, coupon=0.0):
        base = subtotal(items)
        return discount(base, coupon)
    """
)

_CART_TEST = _py(
    """
    from pricing.cart import total
    from pricing.ledger import subtotal


    def test_subtotal_has_no_phantom_unit():
        items = [{"price": 10, "qty": 2}, {"price": 5, "qty": 4}]
        assert subtotal(items) == 40


    def test_total_applies_discount_once():
        items = [{"price": 10, "qty": 2}, {"price": 5, "qty": 4}]
        assert round(total(items, coupon=0.10), 6) == 36.0
    """
)

_CLOCK_BUG = _py(
    """
    class Clock:
        def __init__(self, start=0):
            self.value = start

        def now(self):
            return self.value

        def advance(self, seconds):
            self.value -= seconds
    """
)

_CLOCK_FIX = _py(
    """
    class Clock:
        def __init__(self, start=0):
            self.value = start

        def now(self):
            return self.value

        def advance(self, seconds):
            self.value += seconds
    """
)

_LIMITER_BUG = _py(
    """
    from .clock import Clock


    class TokenBucket:
        def __init__(self, rate, capacity, clock=None):
            self.rate = float(rate)
            self.capacity = float(capacity)
            self.clock = clock or Clock()
            self.tokens = float(capacity)
            self.last = self.clock.now()

        def allow(self):
            now = self.clock.now()
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last = now
            if self.tokens <= 1:
                return False
            self.tokens -= 1
            return True
    """
)

_LIMITER_FIX = _py(
    """
    from .clock import Clock


    class TokenBucket:
        def __init__(self, rate, capacity, clock=None):
            self.rate = float(rate)
            self.capacity = float(capacity)
            self.clock = clock or Clock()
            self.tokens = float(capacity)
            self.last = self.clock.now()

        def allow(self):
            now = self.clock.now()
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last = now
            if self.tokens < 1:
                return False
            self.tokens -= 1
            return True
    """
)

_LIMITER_TEST = _py(
    """
    from ratelimit.clock import Clock
    from ratelimit.limiter import TokenBucket


    def test_clock_advances_forward():
        clock = Clock(0)
        clock.advance(1)
        assert clock.now() == 1


    def test_bucket_allows_capacity_then_refuses_empty_bucket():
        clock = Clock(0)
        bucket = TokenBucket(rate=1.0, capacity=2.0, clock=clock)
        assert bucket.allow() is True
        clock.advance(1)
        assert bucket.allow() is True
        assert bucket.allow() is True
        assert bucket.allow() is False
    """
)

_STORE_BUG = _py(
    """
    class Store:
        def __init__(self):
            self.seen = set()

        def seen_before(self, key):
            return key not in self.seen

        def remember(self, key):
            self.seen.add(key)
    """
)

_STORE_FIX = _py(
    """
    class Store:
        def __init__(self):
            self.seen = set()

        def seen_before(self, key):
            return key in self.seen

        def remember(self, key):
            self.seen.add(key)
    """
)

_WORKER_BUG = _py(
    """
    def process(job, store):
        if store.seen_before(job["key"]):
            return None
        return job["value"] * 2
    """
)

_WORKER_FIX = _py(
    """
    def process(job, store):
        if store.seen_before(job["key"]):
            return None
        store.remember(job["key"])
        return job["value"] * 2
    """
)

_WORKER_TEST = _py(
    """
    from jobs.store import Store
    from jobs.worker import process


    def test_duplicate_job_is_processed_once():
        store = Store()
        job = {"key": "alpha", "value": 3}
        assert process(dict(job), store) == 6
        assert process(dict(job), store) is None
    """
)


def builtin_fixtures() -> tuple[BugFixture, ...]:
    """Return the three deterministic built-in repair repositories."""
    cart = BugFixture(
        id="offbyone_cart",
        task="Fix the multi-module pricing subtotal and coupon calculation",
        files=_fixture_files(
            package="pricing",
            sources={
                "pricing/ledger.py": _CART_LEDGER_BUG,
                "pricing/cart.py": _CART_BUG,
            },
            test_name="test_pricing.py",
            test_source=_CART_TEST,
        ),
        candidates=(
            PatchCandidate(
                name="cart_only",
                edits=(FileEdit("pricing/cart.py", _CART_CART_ONLY_FIX),),
                rationale="remove the extra unit from total only",
            ),
            PatchCandidate(
                name="subtotal_and_cart",
                edits=(
                    FileEdit("pricing/ledger.py", _CART_LEDGER_FIX),
                    FileEdit("pricing/cart.py", _CART_CART_ONLY_FIX),
                ),
                rationale="remove both phantom units across ledger and cart",
            ),
            PatchCandidate(
                name="subtotal_and_cart_retry",
                edits=(
                    FileEdit("pricing/ledger.py", _CART_LEDGER_FIX),
                    FileEdit("pricing/cart.py", _CART_CART_ONLY_FIX),
                ),
                rationale="retry the verified cross-module fix",
            ),
        ),
        probe_hint="assert subtotal(items) == 40 and total(items, coupon=0.10) == 36.0",
    )
    limiter = BugFixture(
        id="token_bucket_clock",
        task="Repair clock advancement and token-bucket boundary handling together",
        files=_fixture_files(
            package="ratelimit",
            sources={
                "ratelimit/clock.py": _CLOCK_BUG,
                "ratelimit/limiter.py": _LIMITER_BUG,
            },
            test_name="test_limiter.py",
            test_source=_LIMITER_TEST,
        ),
        candidates=(
            PatchCandidate(
                name="clock_only",
                edits=(FileEdit("ratelimit/clock.py", _CLOCK_FIX),),
                rationale="make the clock advance forward only",
            ),
            PatchCandidate(
                name="clock_and_bucket",
                edits=(
                    FileEdit("ratelimit/clock.py", _CLOCK_FIX),
                    FileEdit("ratelimit/limiter.py", _LIMITER_FIX),
                ),
                rationale="fix forward time and the one-token boundary",
            ),
            PatchCandidate(
                name="clock_and_bucket_retry",
                edits=(
                    FileEdit("ratelimit/clock.py", _CLOCK_FIX),
                    FileEdit("ratelimit/limiter.py", _LIMITER_FIX),
                ),
                rationale="retry the verified time/bucket fix",
            ),
        ),
        probe_hint="assert clock.now() == 1 and bucket capacity is spent exactly once",
    )
    worker = BugFixture(
        id="idempotent_worker",
        task="Make duplicate queue jobs idempotent across store and worker",
        files=_fixture_files(
            package="jobs",
            sources={
                "jobs/store.py": _STORE_BUG,
                "jobs/worker.py": _WORKER_BUG,
            },
            test_name="test_worker.py",
            test_source=_WORKER_TEST,
        ),
        candidates=(
            PatchCandidate(
                name="store_only",
                edits=(FileEdit("jobs/store.py", _STORE_FIX),),
                rationale="correct the inverted seen predicate only",
            ),
            PatchCandidate(
                name="store_and_worker",
                edits=(
                    FileEdit("jobs/store.py", _STORE_FIX),
                    FileEdit("jobs/worker.py", _WORKER_FIX),
                ),
                rationale="correct seen-before and remember successful jobs",
            ),
            PatchCandidate(
                name="store_and_worker_retry",
                edits=(
                    FileEdit("jobs/store.py", _STORE_FIX),
                    FileEdit("jobs/worker.py", _WORKER_FIX),
                ),
                rationale="retry the verified idempotency fix",
            ),
        ),
        probe_hint="assert duplicate jobs are processed once",
    )
    return (cart, limiter, worker)


_INVOICE_TAX_BUG = _py(
    """
    def tax(amount, rate):
        return amount * rate
    """
)

_INVOICE_TAX_FIX = _py(
    """
    def tax(amount, rate):
        return amount * (rate / 100.0)
    """
)

_INVOICE_LEDGER_BUG = _py(
    """
    def subtotal(items):
        total = 1
        for item in items:
            total += item["price"] * item["qty"]
        return total
    """
)

_INVOICE_LEDGER_FIX = _py(
    """
    def subtotal(items):
        total = 0
        for item in items:
            total += item["price"] * item["qty"]
        return total
    """
)

_INVOICE_PIPELINE_BUG = _py(
    """
    from .ledger import subtotal
    from .tax import tax


    def charge(items, *, rate=10.0):
        base = subtotal(items)
        return base + tax(base, rate) + 1
    """
)

_INVOICE_PIPELINE_FIX = _py(
    """
    from .ledger import subtotal
    from .tax import tax


    def charge(items, *, rate=10.0):
        base = subtotal(items)
        return base + tax(base, rate)
    """
)

_INVOICE_PIPELINE_REPLAY = _py(
    """
    # Inverse replay basis: ledger then tax then pipeline, with no hidden fee.
    from .ledger import subtotal
    from .tax import tax


    def charge(items, *, rate=10.0):
        base = subtotal(items)
        return base + tax(base, rate)
    """
)

_INVOICE_TEST = _py(
    """
    from billing.ledger import subtotal
    from billing.pipeline import charge
    from billing.tax import tax


    def test_invoice_is_ledger_plus_percentage_tax():
        items = [{"price": 10, "qty": 2}, {"price": 5, "qty": 4}]
        assert subtotal(items) == 40
        assert tax(100, 10) == 10
        assert round(charge(items, rate=10), 6) == 44
    """
)

_SCHEDULER_CLOCK_BUG = _py(
    """
    class Clock:
        def __init__(self, start=0):
            self.value = start

        def now(self):
            return self.value

        def advance(self, seconds):
            self.value -= seconds
    """
)

_SCHEDULER_CLOCK_FIX = _py(
    """
    class Clock:
        def __init__(self, start=0):
            self.value = start

        def now(self):
            return self.value

        def advance(self, seconds):
            self.value += seconds
    """
)

_SCHEDULER_QUEUE_BUG = _py(
    """
    class Queue:
        def __init__(self):
            self.seen = set()

        def seen_before(self, key):
            return key not in self.seen

        def remember(self, key):
            self.seen.add(key)
    """
)

_SCHEDULER_QUEUE_FIX = _py(
    """
    class Queue:
        def __init__(self):
            self.seen = set()

        def seen_before(self, key):
            return key in self.seen

        def remember(self, key):
            self.seen.add(key)
    """
)

_SCHEDULER_RUNNER_BUG = _py(
    """
    from .clock import Clock
    from .queue import Queue


    def run(jobs, clock=None):
        clock = clock or Clock()
        queue = Queue()
        output = []
        for job in jobs:
            if not queue.seen_before(job["key"]):
                output.append(job["value"] * 2)
            clock.advance(1)
        return output, clock.now()
    """
)

_SCHEDULER_RUNNER_FIX = _py(
    """
    from .clock import Clock
    from .queue import Queue


    def run(jobs, clock=None):
        clock = clock or Clock()
        queue = Queue()
        output = []
        for job in jobs:
            if not queue.seen_before(job["key"]):
                output.append(job["value"] * 2)
                queue.remember(job["key"])
            clock.advance(1)
        return output, clock.now()
    """
)

_SCHEDULER_RUNNER_REPLAY = _py(
    """
    # Inverse replay basis: lock clock, queue, and runner in test order.
    from .clock import Clock
    from .queue import Queue


    def run(jobs, clock=None):
        clock = clock or Clock()
        queue = Queue()
        output = []
        for job in jobs:
            if not queue.seen_before(job["key"]):
                output.append(job["value"] * 2)
                queue.remember(job["key"])
            clock.advance(1)
        return output, clock.now()
    """
)

_SCHEDULER_TEST = _py(
    """
    from scheduler.runner import run


    def test_duplicates_are_processed_once_with_forward_clock():
        jobs = [{"key": "alpha", "value": 3}, {"key": "alpha", "value": 4}]
        output, now = run(jobs)
        assert output == [6]
        assert now == 2
    """
)


def tangled_fixtures() -> tuple[BugFixture, ...]:
    """Return deeper cross-module fixtures for the branch-racing A/B arm."""
    invoice = BugFixture(
        id="invoice_pipeline",
        task="Repair a tangled ledger, percentage-tax, and pipeline fee calculation",
        files=_fixture_files(
            package="billing",
            sources={
                "billing/tax.py": _INVOICE_TAX_BUG,
                "billing/ledger.py": _INVOICE_LEDGER_BUG,
                "billing/pipeline.py": _INVOICE_PIPELINE_BUG,
            },
            test_name="test_invoice.py",
            test_source=_INVOICE_TEST,
        ),
        candidates=(
            PatchCandidate(
                name="pipeline_only",
                edits=(FileEdit("billing/pipeline.py", _INVOICE_PIPELINE_FIX),),
                rationale="remove the hidden pipeline fee only",
            ),
            PatchCandidate(
                name="ledger_and_pipeline",
                edits=(
                    FileEdit("billing/ledger.py", _INVOICE_LEDGER_FIX),
                    FileEdit("billing/pipeline.py", _INVOICE_PIPELINE_FIX),
                ),
                rationale="cut ledger and pipeline but leave tax unscaled",
            ),
            PatchCandidate(
                name="full_dependency_cut",
                edits=(
                    FileEdit("billing/tax.py", _INVOICE_TAX_FIX),
                    FileEdit("billing/ledger.py", _INVOICE_LEDGER_FIX),
                    FileEdit("billing/pipeline.py", _INVOICE_PIPELINE_FIX),
                ),
                rationale="repair all three coupled modules",
            ),
            PatchCandidate(
                name="full_inverse_replay",
                edits=(
                    FileEdit("billing/tax.py", _INVOICE_TAX_FIX),
                    FileEdit("billing/ledger.py", _INVOICE_LEDGER_FIX),
                    FileEdit("billing/pipeline.py", _INVOICE_PIPELINE_REPLAY),
                ),
                rationale="repair all modules through a higher-cost inverse replay basis",
            ),
        ),
        probe_hint="assert subtotal(items) == 40 and charge(items, rate=10) == 44",
        fix_index=2,
    )
    scheduler = BugFixture(
        id="event_scheduler",
        task="Repair clock direction, seen-before logic, and runner idempotency together",
        files=_fixture_files(
            package="scheduler",
            sources={
                "scheduler/clock.py": _SCHEDULER_CLOCK_BUG,
                "scheduler/queue.py": _SCHEDULER_QUEUE_BUG,
                "scheduler/runner.py": _SCHEDULER_RUNNER_BUG,
            },
            test_name="test_scheduler.py",
            test_source=_SCHEDULER_TEST,
        ),
        candidates=(
            PatchCandidate(
                name="clock_only",
                edits=(FileEdit("scheduler/clock.py", _SCHEDULER_CLOCK_FIX),),
                rationale="advance time forward only",
            ),
            PatchCandidate(
                name="clock_and_queue",
                edits=(
                    FileEdit("scheduler/clock.py", _SCHEDULER_CLOCK_FIX),
                    FileEdit("scheduler/queue.py", _SCHEDULER_QUEUE_FIX),
                ),
                rationale="lock clock and queue but leave the runner unguarded",
            ),
            PatchCandidate(
                name="full_test_locking_order",
                edits=(
                    FileEdit("scheduler/clock.py", _SCHEDULER_CLOCK_FIX),
                    FileEdit("scheduler/queue.py", _SCHEDULER_QUEUE_FIX),
                    FileEdit("scheduler/runner.py", _SCHEDULER_RUNNER_FIX),
                ),
                rationale="repair clock, queue, and runner in test-locking order",
            ),
            PatchCandidate(
                name="full_inverse_replay",
                edits=(
                    FileEdit("scheduler/clock.py", _SCHEDULER_CLOCK_FIX),
                    FileEdit("scheduler/queue.py", _SCHEDULER_QUEUE_FIX),
                    FileEdit("scheduler/runner.py", _SCHEDULER_RUNNER_REPLAY),
                ),
                rationale="repair all modules through a higher-cost inverse replay basis",
            ),
        ),
        probe_hint="assert duplicate jobs are processed once and clock.now() == 2",
        fix_index=2,
    )
    return (invoice, scheduler)


def hard_stall_fixtures() -> tuple[BugFixture, ...]:
    """Three-module reservation invariants with plausible one-sided repairs.

    Candidates are available to every offline arm; live prompts never include
    candidate patches. These are stress fixtures, not SWE-bench claims.
    """
    result = []
    for package, scale in (("inventory", 1), ("wallet", 100), ("quota", 10)):
        state = "def available(total, held):\n    return total\n"
        reserve = "def reserve(total, held, amount):\n    return held + amount\n"
        service = (
            f"from {package}.state import available\n"
            f"from {package}.reserve import reserve\n"
            "def transact(total, held, amount):\n"
            "    updated = reserve(total, held, amount)\n"
            "    return available(total, updated), updated\n"
        )
        fixed_state = "def available(total, held):\n    return total - held\n"
        fixed_reserve = (
            "def reserve(total, held, amount):\n"
            "    if amount < 0 or held + amount > total:\n"
            "        raise ValueError('invalid reservation')\n"
            "    return held + amount\n"
        )
        sources = {
            f"{package}/state.py": state,
            f"{package}/reserve.py": reserve,
            f"{package}/service.py": service,
        }
        tests = (
            f"import pytest\nfrom {package}.service import transact\n"
            f"def test_existing_hold():\n    assert transact({10 * scale}, {2 * scale}, {3 * scale}) == ({5 * scale}, {5 * scale})\n"
            f"def test_full():\n    assert transact({10 * scale}, 0, {10 * scale}) == (0, {10 * scale})\n"
            f"@pytest.mark.parametrize('amount', [-1, {20 * scale}])\n"
            f"def test_invalid(amount):\n    with pytest.raises(ValueError):\n        transact({10 * scale}, 0, amount)\n"
        )
        partial = PatchCandidate(
            "projection_only",
            (FileEdit(f"{package}/state.py", fixed_state),),
            "Repair the visible balance projection",
        )
        validation = PatchCandidate(
            "validation_only",
            (FileEdit(f"{package}/reserve.py", fixed_reserve),),
            "Enforce reservation boundaries",
        )
        complete = PatchCandidate(
            "joint_invariant",
            (*partial.edits, *validation.edits),
            "Preserve projection and mutation invariants together",
        )
        result.append(
            BugFixture(
                id=f"hard_{package}",
                task="Repair reservation accounting and invalid-operation behavior.",
                files=_fixture_files(
                    package=package, sources=sources, test_name="test_reservation.py", test_source=tests
                ),
                candidates=(partial, complete, validation),
                probe_hint="",
                fix_index=1,
            )
        )
    return tuple(result)
