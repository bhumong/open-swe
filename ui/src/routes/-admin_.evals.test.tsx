/** @vitest-environment jsdom */

import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, expect, it } from "vitest"

import type { ReviewerEvalStatus } from "@/lib/api"
import { ReviewerEvalStatusView } from "@/routes/admin_.evals"

afterEach(cleanup)

it("links the completed evaluation to its Langfuse project run", () => {
  const data = {
    name: "reviewer_eval",
    status: "completed",
    langfuse_project: "Open SWE Local",
    limit: null,
    started_at: null,
    finished_at: null,
    created_by: "samuel",
    pid: null,
    exit_code: 0,
    experiment_url:
      "http://localhost:3030/project/open-swe-local/datasets/reviewer/runs/run-1",
    error: null,
    log_tail: null,
    progress: { completed: 2, total: 2 },
    worker_id: null,
    trigger: "local",
    updated_at: "2026-10-20T00:00:00Z",
  } satisfies ReviewerEvalStatus

  render(<ReviewerEvalStatusView data={data} />)

  expect(screen.getByText("Open SWE Local")).toBeTruthy()
  expect(
    screen
      .getByRole("link", { name: "View experiment in Langfuse" })
      .getAttribute("href")
  ).toBe(data.experiment_url)
})
