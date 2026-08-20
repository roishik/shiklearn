"""
suite.py — a Suite is a list of Tasks, run for real against one provider,
producing a SuiteResult. This is the harness's top-level object: `Suite`
+ `SuiteResult` are the "eval harness" named in the brief.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.providers.llm.base import LLMProvider
from evals.runner import run_trial
from evals.types import SuiteResult, Task, TaskResult


@dataclass
class Suite:
    tasks: list[Task]
    name: str = "airport investment agent eval suite"

    def filter(self, category: str | None = None, id_substring: str | None = None) -> "Suite":
        tasks = self.tasks
        if category:
            tasks = [t for t in tasks if t.category == category]
        if id_substring:
            tasks = [t for t in tasks if id_substring in t.id]
        return Suite(tasks=tasks, name=self.name)

    def run(self, provider: LLMProvider, trials_override: int | None = None) -> SuiteResult:
        """Run every task in this suite against `provider`, for real.
        Each task's Trials are isolated from each other and from every
        other task's trials — see evals/runner.py:run_trial."""
        task_results: list[TaskResult] = []
        for task in self.tasks:
            n = trials_override if trials_override is not None else task.num_trials
            trial_results = [run_trial(task, i, provider) for i in range(n)]
            task_results.append(TaskResult(task=task, trial_results=trial_results))
        return SuiteResult(task_results=task_results, provider_name=provider.name, model_name=provider.model)
