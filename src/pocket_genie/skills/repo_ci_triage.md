---
name: repo-ci-triage
description: Protocol for diagnosing failing pytest runs, inspecting git diffs, and isolating regressions.
recommended_tier: t2
tags: [devops, pytest, git, ci]
---

# Repository & CI Failure Triage Playbook

1. **Inspect Working Tree & Recent Commits**
   - Query `devops.git_status` and `devops.git_diff` to identify modified files before running full test suites.
2. **Scoped Test Execution**
   - Run `devops.run_pytest` targeted at the affected test module first to minimize output noise.
3. **Root-Cause Isolation**
   - Read only the relevant line range via `devops.read_file_chunk` rather than loading entire source files into context.
4. **Patch & HITL Gate**
   - Propose minimal diffs via `devops.apply_patch` and wait for Human-in-the-Loop approval before committing.
