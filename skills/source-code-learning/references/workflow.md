# Repository Learning Workflow

Use this reference to select the next useful learning step and produce consistent lessons.

## Phase 0: Scope and revision

Record:

- repository root and revision/branch;
- the user's objective: operate, modify, reproduce, research, or interview-level understanding;
- the primary runtime or product path;
- explicit exclusions for the first pass;
- whether the task is explanation-only or also authorizes documentation changes.

Do not promise a full-repository explanation. Define what “core source complete” means for this project.

## Phase 1: Repository census

Inspect cheaply before reading deeply:

```text
README / project docs
AGENTS or repository instructions
top-level directories
build/package manifests
configuration files
main entrypoints
tests that express contracts
current revision and working state
```

Produce a responsibility table:

| Area | Responsibility | Read now? | Question it answers |
|---|---|---:|---|

Mark generated code, vendored dependencies, large assets, and unrelated backends as later/skip.

## Phase 2: One-page mental model

Before internals, explain:

```text
external input
  -> adaptation/validation
  -> core runtime
  -> main transformation or decision
  -> output adaptation
  -> external effect
```

Name the major modules on each arrow. Use one compact flowchart when there are at least three dependent stages.

Completion check: the learner can say what enters the system, what leaves it, and which component owns each major transition.

## Phase 3: First end-to-end path

Choose the most representative path, usually inference/read/request handling before training/write/background infrastructure.

For every function on the path, capture:

```text
who calls it
input names, semantics, shapes/types
logical transformations
output names, semantics, shapes/types
next consumer
```

Stay at orchestration level first. Treat large encoders, database engines, framework runtimes, or networking stacks as named black boxes until the path is complete.

## Phase 4: Data contracts

Create a transformation table. Separate physical/business dimensions from padded/model/storage dimensions.

| Stage | Field | Shape/type | Semantic change | Code owner |
|---|---|---:|---|---|

Always distinguish:

- renaming from reshaping;
- normalization from discretization/tokenization;
- padding from real dimensions;
- batching from sequence length;
- masks from data values;
- internal outputs from externally executable outputs.

## Phase 5: Core internals

Enter internals in dependency order, not file order. For each mechanism:

1. Explain the prerequisite concept in plain language.
2. Show the smallest relevant formula or pseudocode.
3. Locate the exact implementation block.
4. Trace one concrete example through it.
5. Reconnect it to the end-to-end path.

If the learner is blocked by a prerequisite, pause the module and teach that prerequisite; do not stack undefined concepts.

## Phase 6: Training, write, or state-change path

After the read/inference path, trace how the system learns or mutates state:

```text
dataset/request
  -> target or validation
  -> loss/change calculation
  -> gradients/transaction/update
  -> persisted state/checkpoint/database
```

Identify where local per-item results become a scalar/commit, which state must be restored for strict resumption, and which parameters or records are frozen/excluded.

## Phase 7: Runtime engineering

Only after the core behavior is clear, cover:

- configuration and dependency injection;
- parallelism/sharding/concurrency;
- caching;
- logging and metrics;
- checkpoints/persistence;
- serving/deployment;
- evaluation and tests.

Explain what each mechanism changes: correctness, latency, memory, reproducibility, or observability.

## Guided lesson template

```markdown
# Lesson: subsystem name

## Outcome and architectural position

## Call chain
caller -> function -> callee

## Inputs and outputs
| Variable | Meaning | Shape/type |

## Source blocks
### Block 1: purpose
- code
- shape/semantic change
- why it exists

## Framework/paper connection

## Concrete example

## Recap

## Check yourself
1.
2.
3.

## Natural next step
```

## Persistent learning-note template

```markdown
## Current question

## Verified revision

## Call chain

## Function contracts

## Shape/data table

## Confirmed facts

## Temporary black boxes

## Unresolved questions

## My explanation without looking at code
```

When documenting in a repository, rewrite lessons as durable reference material. Remove chat phrasing, local-only paths, stale assumptions, and unverified claims. Add an index and link from existing project documentation.

## Completion criteria

Core source learning is complete when the learner can, without opening the code:

- draw the primary runtime and state-change paths;
- identify the owner of every major transformation;
- state critical shapes/types and distinguish real versus padded dimensions;
- explain the central algorithm and why its design choices exist;
- describe how state/parameters are updated and persisted;
- name important project boundaries and unsupported claims;
- know where to resume when a new question appears.

Deeper framework internals, alternate backends, extensions, and third-party code can remain separate advanced stages.
