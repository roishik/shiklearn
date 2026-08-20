"""
evals/ — a small, real, runnable eval harness for app/agent_loop.py's
sample agent.

Implements Anthropic's published agent-eval anatomy as actual Python
types, not informally-structured dicts:

  Task    — inputs + success criteria (evals/types.py:Task)
  Trial   — one attempt at a task; run several per task, agents are
            non-deterministic (evals/types.py:Trial)
  Trace   — the transcript: tool calls, reasoning, intermediates
            (evals/types.py:Trace, wraps app.agent_loop.AgentResult)
  Outcome — the final environment state derived from a Trace: which
            tools were called, with what arguments, what numbers appear
            where (evals/types.py:Outcome)
  Grader  — an assertion over (Task, Outcome) that returns partial
            credit, not just pass/fail (evals/types.py:Grader,
            GradeResult; concrete graders in evals/graders/)
  Suite   — a collection of Tasks, run for real against a provider,
            aggregated into a report (evals/suite.py:Suite, SuiteResult)

See evals/README.md for how to add a task in under 2 minutes and how to
read a suite's report.
"""
