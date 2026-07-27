"""
Evaluation runner.

Writes one JSONL record per (level, task, repetition). Repetitions at
temperature > 0 are not optional: seed-consistency is one of the strongest
detectors, and you cannot compute it from a single pass.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, List, Optional

from .backends import Completion, MockSandbagger
from .prompts import REFERENCE_WEAK_PROMPT, build_prompt
from .tasks import Task, build_bank, grade_answer, extract_answer


@dataclass
class Record:
    level: int
    tid: str
    domain: str
    family: str
    difficulty: int
    kind: str
    rep: int
    correct: bool
    extracted: str
    gold: str
    reasoning: str
    answer_text: str
    latency_s: float
    n_tokens: int
    unlocked: bool = False


def run_sweep(backend, *, levels: Iterable[int], bank: Optional[List[Task]] = None,
              reps: int = 3, temperature: float = 0.7, out_path: str = "results/raw.jsonl",
              password: Optional[str] = None, progress: bool = True,
              max_tokens: int = 1024) -> List[Record]:
    bank = bank if bank is not None else build_bank()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    records: List[Record] = []
    is_mock = isinstance(backend, MockSandbagger)

    with open(out_path, "w") as fh:
        for level in levels:
            if is_mock:
                backend = MockSandbagger(level, seed=backend.seed,
                                         honest_accuracy=backend.honest_accuracy)
            system = build_prompt(level, password=password) if level != -1 \
                else REFERENCE_WEAK_PROMPT
            for rep in range(reps):
                for task in bank:
                    if is_mock:
                        comp = backend.generate_for_task(task, rep=rep)
                    else:
                        comp = backend.generate(
                            system, task.prompt,
                            temperature=temperature if reps > 1 else 0.0,
                            max_tokens=max_tokens,
                            seed=None if reps > 1 else 0)
                    rec = Record(
                        level=level, tid=task.tid, domain=task.domain,
                        family=task.family, difficulty=task.difficulty,
                        kind=task.kind, rep=rep,
                        correct=grade_answer(task, comp.full),
                        extracted=extract_answer(comp.full),
                        gold=task.answer, reasoning=comp.reasoning,
                        answer_text=comp.text, latency_s=comp.latency_s,
                        n_tokens=comp.n_tokens)
                    records.append(rec)
                    fh.write(json.dumps(asdict(rec)) + "\n")
                if progress:
                    print(f"  level {level:>3} rep {rep}: "
                          f"{len(bank)} items done", flush=True)
            fh.flush()
    return records


def run_password_probe(backend, *, level: int, password: str,
                       bank: Optional[List[Task]] = None, reps: int = 1,
                       out_path: str = "results/password.jsonl") -> List[Record]:
    """Same items with and without the unlock token.

    A within-model capability ceiling is much stronger evidence of deliberate
    suppression than any cross-model comparison, because it holds the weights,
    the prompt and the items fixed.
    """
    bank = bank if bank is not None else build_bank()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    system = build_prompt(level, password=password)
    out: List[Record] = []
    with open(out_path, "w") as fh:
        for unlocked in (False, True):
            for rep in range(reps):
                for task in bank:
                    user = task.prompt + (f"\n{password}" if unlocked else "")
                    comp = backend.generate(system, user, temperature=0.0)
                    rec = Record(level=level, tid=task.tid, domain=task.domain,
                                 family=task.family, difficulty=task.difficulty,
                                 kind=task.kind, rep=rep,
                                 correct=grade_answer(task, comp.full),
                                 extracted=extract_answer(comp.full),
                                 gold=task.answer, reasoning=comp.reasoning,
                                 answer_text=comp.text,
                                 latency_s=comp.latency_s,
                                 n_tokens=comp.n_tokens, unlocked=unlocked)
                    out.append(rec)
                    fh.write(json.dumps(asdict(rec)) + "\n")
    return out


def load_records(path: str) -> List[Record]:
    with open(path) as fh:
        return [Record(**json.loads(ln)) for ln in fh if ln.strip()]
