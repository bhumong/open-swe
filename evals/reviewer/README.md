# Reviewer Eval

Offline evaluation of the Open SWE Reviewer graph against the 50 PRs and 173
categorized golden comments from `withmartian/code-review-benchmark`
(`offline/golden_comments` at `e616e84`, the 2026-08 golden refresh). The
checked-in `golden_comments/` files remain the canonical benchmark source.

## Layout

```
evals/reviewer/
├── golden_comments/      # 50 PRs × golden comments
├── build_dataset.py      # martian JSON → repeatably synced Langfuse dataset
├── config.toml           # default benchmark run config
├── costs.py              # per-PR cost/token scores from reviewer traces
├── judge.py              # claude-opus-4-5 pairwise match evaluator + aggregate
├── target.py             # invokes the reviewer graph over langgraph_sdk
├── store_reporter.py     # publishes live progress to the dashboard store record
└── run_eval.py           # Langfuse experiment SDK entrypoint
```

## Prerequisites

- Python 3.14 or newer and `uv`.
- `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` for the Langfuse project that
  owns the benchmark dataset and experiment traces. `LANGFUSE_BASE_URL` defaults
  to Langfuse Cloud and can point to a self-hosted instance.
- `gh` authenticated (`gh auth status`) to resolve current PR SHAs when syncing
  the dataset.
- A LangSmith key with `gateway:invoke` in `LANGSMITH_GATEWAY_API_KEY` (or
  `LANGSMITH_API_KEY`) if the judge uses the retained LangSmith LLM Gateway.
- A running reviewer graph (`langgraph dev` or a deployed assistant) with
  `REVIEWER_ASSISTANT_ID` pointing at it. Defaults to assistant `reviewer` on
  `http://localhost:2024`.

## 1. Sync the dataset

```bash
# Dry run — writes evals/reviewer/dataset_dryrun.json without uploading
uv run --python 3.14 python -m evals.reviewer.build_dataset --dry-run

# Create or update the Langfuse dataset
uv run --python 3.14 python -m evals.reviewer.build_dataset \
  --dataset-name openswe-reviewer-v2
```

Each example carries the PR repository, number, URL, title, base/head SHAs and
refs, and golden comments. Stable Langfuse item IDs let repeated syncs update
canonical examples in place; unrelated dataset items are left alone.

## 2. Run the evaluation

The reviewer graph must be running and accept the benchmark message/config
input. Evaluation runs record findings with `add_finding` and finish with
`publish_review`, which persists the exact ordered publication snapshot scored
by the harness.

```bash
uv run --python 3.14 python -m evals.reviewer.run_eval
```

Smoke-test with 3 PRs first:

```bash
uv run --python 3.14 python -m evals.reviewer.run_eval --limit 3
```

The Langfuse Python experiment SDK creates one dataset run and trace per item,
attaches the judge and cost scores, and exposes the result URL in its output.
The benchmark runs only when explicitly started.

### From the dashboard

Admins start a run from **Evals** in the sidebar (`/admin/evals`). Under
`langgraph dev`, Open SWE launches the evaluator as a local child process using
the current checkout and local `.env`. A deployed instance retains the existing
LangSmith sandbox worker for compute; it injects Langfuse authorization through
a host-scoped proxy rule without putting real Langfuse keys in the worker
environment. Both modes publish progress, logs, and the Langfuse experiment link
to the same dashboard status record.

## Scoring

The judge prompt and model (`claude-opus-4-5`) match upstream. Unprefixed
metrics count every golden (upstream's "all" profile). `strict_*` and `core_*`
follow upstream's category profiles: a candidate that matches a golden outside
the profile is neither a true nor a false positive, and only in-profile goldens
count as misses. Upstream's leaderboard headline is **`core_micro_f2`**.

| Profile | Categories | Goldens |
|---|---|---:|
| `strict` | bug, security, concurrency, data, api | 139 |
| `core` | strict + perf, test_gap, doc_defect | 158 |
| all (unprefixed) | core + style, speculative | 173 |

Upstream also runs an LLM step that groups duplicate comments before judging.
It is not ported: the reviewer's published findings are already deduplicated,
and the judge drops exact duplicates.

By default the judge scores the exact final `surfaced_findings` snapshot,
including only renderable findings selected by `publish_review`. Set
`score_mode = "all_findings"` only to diagnose deduplicated `add_finding` calls
before publication.

`model_id` and `reasoning_effort` in `config.toml` configure the reviewer run,
so isolated benchmark deployments can test a model without changing the
deployment-wide default.

## Costs

Each target returns its reviewer thread and invocation ID. The cost evaluator
uses those exact trace metadata values to read generation costs and usage from
Langfuse, then attaches `cost_usd` and `total_tokens` scores to the experiment
item. Missing or incomplete prices are marked `cost_status=unpriced`; they are
not recorded as zero. Aggregate cost, token, priced-item, and unpriced-item
scores are attached to the dataset run. Costs cover LLM calls only, not sandbox
time.
