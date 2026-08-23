---
name: source-code-learning
description: Systematically teach and document an unfamiliar codebase by building a project map, tracing end-to-end call and data flows, explaining code in progressive lessons with shapes and checkpoints, and connecting ML papers to implementation. Use when the user asks to learn, understand, 梳理, 讲解, or study a repository step by step; not for ordinary feature implementation, bug fixing, or one-off code review unless learning is the goal.
---

# Source Code Learning

Help the user form a durable mental model of a repository, not merely understand isolated lines.

## Start from evidence

1. Inspect repository instructions, revision, layout, configuration, tests, and current working state before teaching.
2. Identify the user's learning goal and current depth from the conversation. Make a reasonable assumption if this is already clear; do not repeatedly ask for level confirmation.
3. Pin explanations to the inspected revision. Link local files and line numbers when available, and distinguish verified code behavior from inference or paper-level intent.
4. Treat reading and explanation as read-only. Create notes, edit the repository, or publish material only when the user asks.

## Choose the current learning mode

- **Map mode:** Use when the user lacks whole-project control. Build a one-page architecture, directory responsibility map, entrypoints, and one end-to-end path before opening internals.
- **Guided lesson mode:** Use for “continue/next part.” Teach one coherent subsystem at a time and preserve the established sequence.
- **Question drill-down mode:** Use when the user points to a line or concept. Explain its local effect, upstream inputs, downstream consumers, and why the design exists, then reconnect it to the whole system.
- **Consolidation mode:** Use when asked to整理 or document learning. Turn verified lessons into maintained repository documentation, shape tables, call graphs, and completion checklists.

Read [references/workflow.md](references/workflow.md) for the phase order, lesson contract, note templates, and completion criteria. Read [references/ml-vla.md](references/ml-vla.md) only for machine-learning, robotics, VLA, diffusion/flow, or paper-to-code repositories.

## Teaching contract

For each lesson:

1. Lead with what this subsystem accomplishes and where it sits in the architecture.
2. Show the caller → current function → callee chain.
3. Group source into logical blocks; do not paraphrase every line independently.
4. Track important data contracts and tensor shapes through every semantic transformation.
5. Explain prerequisites such as attention, residuals, gradients, or normalization before relying on them.
6. Connect implementation choices to the framework or paper, while noting anything the public code does not implement.
7. Use a small numerical example or compact diagram when it materially clarifies the mechanism.
8. End with a short recap and 3–6 questions the learner should answer without looking at the code.
9. State the natural next subsystem, but do not silently jump ahead of the user's pace.

Match the user's language and technical level. Prefer stable notation across lessons. When the code uses overloaded names, explicitly separate their meanings.

## Preserve the main thread

Keep a visible distinction between:

- the project goal;
- runtime entrypoints;
- data semantics and shapes;
- model/business internals;
- training or write path;
- deployment, evaluation, and engineering infrastructure.

When a framework API obscures the main idea, first translate it to plain pseudocode, then explain the API detail. Mark low-priority modules as temporary black boxes instead of derailing the current path.

## Quality bar

A learning stage is complete only when the user can reconstruct its call chain, name the inputs and outputs, explain the key design choice, and relate it to the project goal. “All files were mentioned” is not completion.
