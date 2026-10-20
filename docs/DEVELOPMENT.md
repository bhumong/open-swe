# Local Development

Run Open SWE on your machine with the backend and dashboard at `http://localhost:2024` and a local Langfuse instance at `http://localhost:3030`. The normal development path stays on loopback and does not need a tunnel, Slack app, or GitHub App. The optional webhook integration path is at the end of this guide. Deploying Open SWE for a team is covered by the [installation guide](INSTALLATION.md).

## Prerequisites

- **Python 3.14+** and [uv](https://docs.astral.sh/uv/)
- [LangGraph CLI](https://docs.langchain.com/langsmith/cli) (installed by `uv sync`)
- Node 22.22.2+ and [pnpm](https://pnpm.io/) for the dashboard
- Docker with the Compose plugin
- [GitHub CLI](https://cli.github.com/) authenticated with `gh auth login` (local dashboard sign-in)
- An Anthropic API key; this checkout can copy it from the parent workspace's `.env`

## 1. Clone and install

```bash
git clone https://github.com/langchain-ai/open-swe.git
cd open-swe
uv venv
source .venv/bin/activate
uv sync --all-extras
```

## 2. Prepare the local environment

Authenticate `gh` if needed, then create the local environment files:

```bash
gh auth login
make local-setup
```

The setup command reads only `ANTHROPIC_API_KEY` and any configured `LANGSMITH_*` values from the parent workspace `.env` (`../../.env` from this nested checkout). Set `OPEN_SWE_SOURCE_ENV` to another env-file path when needed. It creates the Langfuse initialization file at `infra/langfuse/.env` and Open SWE's `.env`, both ignored by Git with owner-only permissions. Generated passwords and API keys are not printed. It sets `LLM_MODEL_ID=anthropic:claude-haiku-5-5`, allows and grants admin access to the `gh` user, disables LangSmith tracing, and keeps configured LangSmith Gateway/Managed Tools settings.

If the repository-root `.env` already exists, the command leaves it untouched. Merge these local settings into it manually. To initialize Langfuse separately, use `make langfuse-setup`.

Open SWE uses the `local` sandbox provider in this profile. It runs shell commands as your host user and is **not an operating-system isolation boundary**. Its workspace is `~/open-swe-local-workspace`; use a dedicated directory and do not place sensitive files there.

## 3. Start local Langfuse

```bash
make langfuse-up
```

This starts Langfuse web, worker, Postgres, ClickHouse, Redis, and MinIO with persistent named volumes. Only the UI/API is published, on `127.0.0.1:3030`; its databases, cache, and object storage have no host ports. Open `http://localhost:3030`. The initial owner login and API keys are already created. To read the login, inspect only these two entries in the private file:

```bash
grep -E '^LANGFUSE_INIT_USER_(EMAIL|PASSWORD)=' infra/langfuse/.env
```

`make langfuse-down` stops the services and preserves their volumes. Do not add `-v` if you want to keep local traces and datasets. This profile does not expose MinIO's media endpoint, so Langfuse media uploads are unavailable; ordinary text traces and evaluations work.

## 4. Run Open SWE

Build the dashboard and start the local LangGraph runtime:

```bash
make build-dashboard
make dev
```

`make dev` starts Open SWE's PostgreSQL container on `127.0.0.1:5433` when `POSTGRES_URI` is unset. This database is separate from Langfuse's internal Postgres. Open SWE and Langfuse both stay local; don't expose port 2024 through a tunnel because `langgraph dev` has unauthenticated runtime APIs.

`langgraph dev` serves the graphs, the FastAPI app, and the dashboard build together on port 2024. Rebuild the dashboard after UI changes, or skip `make build-dashboard` if you only need the API. After changing `.env`, restart the backend.

**Working on the UI?** Use `make dev-ui` for Vite hot reload with the backend still fronting port 2024:

```bash
make dev-ui
```

## 5. Verify the local loopback setup

Sign in through the dashboard's **Continue with GitHub** link, which uses the authenticated `gh` CLI in `langgraph dev`. Then start a simple agent task. Confirm it appears in the local Langfuse project and that its trace link opens the same trace. The Langfuse project and admin evaluation page are available locally; the reviewer benchmark is not run automatically.

GitHub App-backed operations (publishing reviews, inline review comments, and inbound webhooks) are unavailable without a GitHub App. Per-user GitHub reads work through the local `gh` login. For that path, `gh auth login` must be for the same user written to `ALLOWED_GITHUB_USERS` by setup.

## Optional: GitHub and Slack webhooks

Only follow this section when testing inbound webhooks. The loopback workflow above does not need an internet-facing tunnel.

Create a machine-specific GitHub App using [Create a GitHub App](INSTALLATION.md#3-create-a-github-app). Use `http://localhost:2024/dashboard/api/auth/callback` as the callback and install it on the repositories you want to test. For inbound GitHub events, the webhook URL must use your static ngrok domain because GitHub cannot reach localhost.

For a first-time tunnel setup, the free ngrok plan gives you a static domain:

1. Sign up at [dashboard.ngrok.com](https://dashboard.ngrok.com/signup) and install the agent (`brew install ngrok`, or use the download from the dashboard).
2. Connect it with the `ngrok config add-authtoken …` command shown under **Getting Started → Your Authtoken**.
3. Under **Domains**, claim the free static domain, such as `<name>.ngrok-free.dev`.
4. Start the restricted webhook tunnel and leave it running:

   ```bash
   make tunnel NGROK_DOMAIN=<name>.ngrok-free.dev
   ```

`make tunnel` uses [`examples/ngrok/webhooks-only.yml`](../examples/ngrok/webhooks-only.yml), which exposes only `/webhooks/*`. That restriction is required: the LangGraph API itself (`/threads`, `/runs`, `/assistants`, `/store`, …) has no authentication in `langgraph dev`. Keep dashboard and LangGraph routes on `http://localhost:2024`. Once running, `curl https://<name>.ngrok-free.dev/webhooks/slack` should reach the backend; `/ok` should receive ngrok's 404.

For local Slack OAuth, keep the callback relay in the traffic policy: requests to `/dashboard/api/slack/callback` on the ngrok domain must redirect to `http://localhost:2024/dashboard/api/slack/callback` with the full query string. Do not expose dashboard pages or API routes through the tunnel.

Create a separate Slack app for this checkout using [Create the Slack app](INSTALLATION.md#5-create-the-slack-app), with your ngrok domain and a distinct bot name. Add the app's bot token, signing secret, user ID, username, and `SLACK_PUBLIC_BASE_URL` to the ignored root `.env`. If Slack does not verify its events URL, restart `make dev` and press **Retry** in **Event Subscriptions**.

For this optional integration, add the GitHub App's IDs, private key, webhook secret, and installation ID to `.env` too. `LANGGRAPH_URL`, `DASHBOARD_BASE_URL`, and `DASHBOARD_API_BASE_URL` default to localhost. Model routing is documented in [Model providers and API keys](INSTALLATION.md#4-model-providers-and-api-keys); a configured LangSmith Gateway remains available without sending traces there. Linear setup is in the [Linear guide](INSTALLATION.md#linear).

## Local PostgreSQL and transcript storage

Open SWE also needs PostgreSQL for its own tables; `langgraph dev` keeps threads and Store data in memory but does not provide this database. `make dev` and `make dev-ui` start a `postgres:16` container named `open-swe-postgres` on port 5433 when `POSTGRES_URI` is unset. Its `open-swe-postgres` volume preserves users, workspaces, and settings when the container stops. `make postgres` starts it independently; use `docker compose down` to stop it. Set `POSTGRES_URI` to use another database where the app can create schemas.

With this database, new threads are also written to the append-only transcript event log and stamped `transcript: v2`; the dashboard reads recorded threads from this log. Older threads without recorded agent turns continue to read LangGraph state.

`TEST_ANALYTICS_POSTGRES_URI` is only for tests. Tests clone temporary databases from a migrated template and drop them afterwards, so the role needs `CREATEDB`. A separate throwaway `postgres:16` container can use port 5439; unset this variable and the affected tests skip.

## Dashboard and backend commands

`make dev` refuses to start while something else listens on port 2024, and names the process. When switching worktrees, stop the previous backend gracefully and wait for it to release port 2024 before starting the replacement. Prepare the [worktree state](#langgraph-state-across-worktrees) before startup.

`langgraph dev` serves the graphs, the FastAPI app, and the dashboard build together on port 2024. The bundled UI is static, so rebuild it after UI changes, or skip `make build-dashboard` if you only need the API. It reloads on code changes only: after editing `.env`, restart it.

**Working on the UI?** Have the backend front the Vite dev server instead of serving a build:

```bash
make dev-ui   # Vite on :3000 and the backend on :2024 forwarding UI requests to it, in one terminal
```

`make dev-ui` runs `make web` and `make dev` side by side, the backend with `DASHBOARD_DEV_SERVER_URL=http://localhost:3000`; Ctrl-C stops both. Open `http://localhost:2024` as usual: the page, its modules, and hot module replacement come from Vite, while `/dashboard/api/*` and the LangGraph routes stay with the backend. Nothing else changes, because the browser never leaves port 2024. The HMR WebSocket connects straight to Vite's port; the UI's Vite config points the client there.

| Endpoint | Purpose |
|---|---|
| `/` | Dashboard |
| `POST /webhooks/github` | GitHub issue, PR, and comment webhooks |
| `POST /webhooks/slack`, `POST /webhooks/slack/interactivity` | Slack events and Block Kit interactions |
| `POST /webhooks/slack/commands` | The `/oswe` slash command |
| `POST /webhooks/linear` | Linear comment webhooks |
| `GET /dashboard/api/auth/login`, `GET /dashboard/api/auth/callback` | GitHub login |
| `/dashboard/api/*` | Dashboard API |
| `GET /ok`, `GET /health` | Health checks |

> `make run` serves the FastAPI app alone with uvicorn on port 8000, without the LangGraph runtime. Nothing that creates runs works there; use `make dev`.

With only the seeded `default` workspace, runs land there by default. Create additional workspaces from **Workspaces** to exercise routing locally: thread, `/workspace:<slug>` on the opening message (`workspace:<slug>` and `env:<slug>` also work), owning repository, bound Slack channel, user default, then `default`. See [How a run picks its workspace](INSTALLATION.md#7-verify-it-works) in the installation guide. Incidents require the optional Slack integration above.

Record worktree, process IDs, and local state locations in ignored `logs/local-dev/` notes in the primary checkout if you need to resume later.

## Backend API documentation

[`swagger.json`](../swagger.json) is the generated OpenAPI 3.1 schema for the custom FastAPI backend (`openswe.webapp:app`). Import it into an OpenAPI 3.1-compatible viewer. After local setup, `make run` serves interactive documentation at `http://localhost:8000/docs` and the live schema at `/openapi.json`; this server does not include the LangGraph runtime or support creating runs.

Regenerate the checked-in schema with `make swagger` after changing backend routes or models. The checked-in file can lag the running backend; use its live schema when inspecting deployed routes. Some request/response schemas and authentication requirements are not yet documented. LangGraph runtime endpoints such as `/runs`, `/threads`, and `/assistants` are not included.

## LangGraph state across worktrees

`langgraph dev` persists local threads, checkpoints, and Store data under `.langgraph_api` in its working directory. Preserve existing local data when moving development to a new worktree unless you want a clean start.

The repository's [`.worktreeinclude`](../.worktreeinclude) includes the root `.langgraph_api` directory. [Local Codex-managed worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees#copy-ignored-local-files-into-managed-worktrees) copy matching ignored files when they are created. Stop the source backend gracefully so it flushes persistence, and back up its state before creating the worktree. This is a snapshot, not ongoing synchronization.

Codex skips source symlinks and preserves files already present in the destination. Create the worktree from the primary checkout containing the actual state directory. For worktrees created with `git worktree add`, or when the source is a symlink, copy the stopped primary checkout's entire `.langgraph_api` directory manually, including hidden files. Preserve any existing destination state rather than overwriting it automatically.

For one continuous local instance across worktrees, link `.langgraph_api` to the primary checkout's state directory instead of copying it. Only one backend may use that shared directory at a time; restart from the desired worktree to switch code. Independently copied state diverges after creation and also retains schedules and integration settings, so avoid running multiple copies against the same live integrations, which can duplicate background work.

Keep state, backups, and environment files ignored by Git. Reuse the existing local environment, including its token-encryption settings, without printing credentials. `.worktreeinclude` copies local state into worktrees; it does not add that state to version control.

## Sign in locally with the `gh` CLI

The dashboard's per-user reads — the PR list, one PR's details, a PR preview — run on the signed-in person's own OAuth token, and the `gh` CLI already holds one. `GET /dashboard/api/auth/dev-login` stores it and mints the session, so a machine-local GitHub App (see the optional webhook setup above) is only needed for App-backed endpoints. With no `GITHUB_APP_CLIENT_ID` configured, **Continue with GitHub** redirects here on its own; `/dashboard/api/auth/dev-login?redirect_to=/agents/reviews` skips the page.

It is refused with a 404 outside `langgraph dev`, which is the only runtime reporting the `local_dev` API variant, and the `ALLOWED_GITHUB_USERS` allowlist still applies — set it to your own login. `DASHBOARD_JWT_SECRET` signs the session.

What still needs the App, and answers `503 GitHub App token unavailable` without one: a published review and its diff, inline review comment reads and writes, and the webhook flows. The reviews list and each PR's preview read as you, so they work.

The `local-gh-dev` skill in `.claude/skills/` walks through the whole loop, including the port and Postgres conflicts between worktrees.

## Dashboard on the Vite dev server directly

`make dev-ui` is the simple way to develop the UI. Opening Vite on `http://localhost:3000` directly also works:

```bash
pnpm install      # from the repo root: ui/ and desktop/ are one pnpm workspace
make web          # Vite on http://localhost:3000, proxying /dashboard/api/* to DASHBOARD_API_URL (default http://localhost:2024)
```

The browser now talks to `http://localhost:3000`, so the session cookie has to be set on that origin and the login callback has to return there:

```bash
DASHBOARD_BASE_URL="http://localhost:3000"       # the frontend origin; allowed for the CSRF check and post-login redirects
DASHBOARD_API_BASE_URL="http://localhost:3000"   # what browsers use for /dashboard/api/* and the OAuth callback
```

and the GitHub App needs `http://localhost:3000/dashboard/api/auth/callback` as an additional callback URL. Keep both URLs on `http://` locally so the cookie is `SameSite=Lax`. `DASHBOARD_ALLOWED_ORIGINS` lists **additional** origins that may call the API with credentials; credentialed CORS is only enabled when it is set, and `*` is rejected.

## Dashboard against a deployed backend

`pnpm run dev:prod` runs the Vite dev server, hot reload and all, against a deployment instead of a local backend — UI work against real threads, repositories, and settings without running the agent locally. Put the deployment in the gitignored root `.env`:

```bash
DASHBOARD_API_URL=https://your-deployment.example.com   # a preview deploy, not production
```

then `pnpm run dev:prod`. There is no default, because a fallback would be someone else's production. `DASHBOARD_API_URL` in the environment overrides the file, and it is the only key read out of `.env`: the rest of it is the local backend's secrets, which have no business in the environment of a dev server the browser talks to.

The first run opens your browser to sign in. That leg still runs against the deployment's own registered GitHub callback — nothing changes on the GitHub App, and nothing changes on the deployment. It is the PKCE loopback handoff the desktop app uses (`?desktop_handoff=…&desktop_port=…`): the deployment redirects your browser to `http://127.0.0.1:<port>/callback` with a code that lives 120 seconds, and the script exchanges it at `/dashboard/api/auth/desktop/exchange` for a session. That tab then waits for Vite and sends you on to the dev server, so signing in ends where you want to be. The session lasts a week and is cached in `~/.cache/open-swe/dev-session.json`; after that you sign in again.

The dev server then attaches that session to everything it proxies, and presents the deployment's own origin so its CSRF check accepts mutations. **The session is real.** A task started from `http://localhost:3000` is a real run on that deployment, and the Admin pages edit its real settings.

`pnpm run build`, `pnpm run typecheck`, and `pnpm run test` run across the workspace through Turborepo (`pnpm --filter open-swe-dashboard run <script>` scopes one); `pnpm run lint` (oxlint) and `pnpm run format` / `pnpm run format:check` (oxfmt) run once from the root over every JS and TS file.

## Test a PR in preview (LangChain maintainers)

The shared [preview environment](https://open-swe-preview-cc53e8fbe667565d843d0843f84ee92c.us.langgraph.app/agents) combines `main`, `preview-manual`, and open PRs labeled `preview` whose branch lives in this repository (anyone with write access; a fork's code stays out); it is not an isolated deployment per PR. (Note: staging follows `main` and is for post-merge testing.)

1. Add the **`preview`** label to your PR.
2. Run [Deploy open-swe preview](https://github.com/langchain-ai/langchainplus/actions/workflows/deploy_open_swe_preview.yaml) on `main` with **force** unchecked, or wait for a scheduled run at :04, :19, :34, or :49 each hour. Labeling alone does not deploy.
3. In the run summary, confirm your PR and head commit appear under **Preview tree → Merged**, not **Skipped**. A successful run may still omit a PR.
4. Wait for the run's **Deploy the preview tree** job. It rolls the preview deployment to the published `preview` commit and fails, with the revision's build or server logs, unless the revision deploys.
5. Open [preview](https://open-swe-preview-cc53e8fbe667565d843d0843f84ee92c.us.langgraph.app/agents) and test with non-production tasks and repositories. For local UI iteration against that backend, see [Dashboard against a deployed backend](#dashboard-against-a-deployed-backend).

When PRs conflict with the preview tree, the run makes one `oswe` call (an Open SWE agent on the backend in the `OPEN_SWE_BACKEND_URL` repository variable) that merges all of them; the summary marks those PRs `conflicts resolved by oswe`, and later runs replay the resolution from a git rerere cache. PRs the agent cannot resolve are skipped, unlabeled, and receive resolution instructions for the shared `preview-manual` branch. Resolve the conflict before reapplying the label. Use **force** only to rebuild an unchanged preview tree.

Before publishing a changed tree, the run typechecks the dashboard. On failure it reassembles once without the rerere cache, then asks `oswe` to commit a fix-up (`preview: fix typecheck errors (oswe)`), which later runs replay while the tree underneath is unchanged. If the typecheck still fails, nothing is published, the run fails, and `refs/preview-failed` records the tree so later runs fail fast until it changes.

The label stays on through pushes: each push to a labeled PR rebuilds the preview with its new head.

To remove a PR, remove its label and trigger or await another run; the current deployment remains until its replacement deploys, and changes in `main` or `preview-manual` remain. Every Sunday, the first scheduled run between 07:00 and 07:59 `America/New_York` resets preview to `main`, removes labels, and deletes `preview-manual`.

## Desktop app (experimental)

The Electron app in `desktop/` includes the compiled dashboard UI. Run it next to the backend:

```bash
pnpm install                  # from the repo root
make dev                      # terminal 1
pnpm run dev:desktop          # terminal 2
```

Development connects to `http://localhost:2024`. For a hosted backend run `pnpm --dir desktop run start -- --backend-url=https://your-backend.example.com` or set `OPEN_SWE_BACKEND_URL`. `pnpm --dir desktop run pack` creates an unpacked application and `pnpm --dir desktop run dist` an installer. Packaged builds ask for the organization's backend URL on first launch and never default to the maintainers' deployment. The GitHub App must allow `<backend-url>/dashboard/api/auth/callback` for desktop login.

## Profiling thread load and streaming

The dashboard records two performance spans, in every build, with the same code path locally and in production (`ui/src/lib/perf/`):

| Span | Starts | Steps | Ends |
|---|---|---|---|
| `thread_load` | The navigation to `/agents/:threadId` (or the document's time origin on a full page load, `cold=true`) | `detail` (thread summary, `GET /threads/:id`), `hydrate` (SDK state fetch, `GET /threads/:id/state`), `paint` | First frame after the transcript rendered |
| `agent_run` | Pressing send (`joined=true` when the run was started elsewhere, e.g. a queued message) | `accepted`, `stream_open`, `first_event`, `generation_start` (first assistant message, so the run is visibly producing thinking or a tool call), `first_text` (first streamed assistant text) | The run's streaming phase ends (`reason`: success, error, interrupt, stopped) |

Each span carries attributes: request time-to-first-byte and the backend's `Server-Timing` phases for the detail and state requests (`detail_srv_thread_get_ms`, `state_srv_get_state_ms`, …), whether the detail came from the sidebar cache, message and chunk counts, time spent in `streamMessagesToUi` (`build_ms`), protocol event and text-delta counts, the lag between the server's event timestamp and receipt (`lag_avg_ms`, includes clock skew, read as a trend), and, in development builds only, React commit time for the transcript while streaming (`commit_ms`).

**Locally.** Spans print to the console in dev builds (`[perf] thread_load 812ms — detail 120 · hydrate 640 · paint 812`). Open a thread with `?perf=1` for an overlay listing recent spans with a copy-as-JSON button (`?perf=0` hides it again; the flag persists in `localStorage`). `window.__openSwePerf.spans()` and `.export()` return the same data for scripts and Playwright. Every span is also a User Timing mark and measure named `osw:*`, so it appears on the Timings track of the Chrome Performance panel next to long tasks and network requests, which is where to look once a span says *what* is slow. Compare cold and warm loads separately (`cold`, `detail_cached`), and reload a few times per change: single samples are noisy.

**In production.** With Datadog RUM configured, ended spans are sent as custom duration vitals named `thread_load` and `agent_run`, attributes in the vital context and `client` (`web` or `desktop`) in the global context. Abandoned spans (navigated away, hydration failed) stay local only. The backend side of the same picture is the `Server-Timing` header on `GET /dashboard/api/threads/{id}` and `/state` (logged as `thread state timings`) and the `open_swe_dashboard_thread_ttft` histogram, which measures the same thing as `first_text` rather than the first token of any kind.

For "how long until the agent answers", read `@context.step_generation_start_ms`; `@context.step_first_text_ms` sits after the opening tool calls and is much larger.

**Desktop diagnostics.** Installed builds keep *View → Toggle Developer Tools* and add *Help → Save Diagnostics Report…*, which writes the renderer's recent console output, the main process's warnings, app and OS versions, and the exported perf spans to a text file, with session cookies, bearer tokens and provider keys redacted. Ask users to attach that file to a report.

## Make targets

| Target | What it does |
|---|---|
| `make local-setup` | Creates private, ignored local Langfuse and Open SWE env files |
| `make langfuse-setup` | Creates or reuses only the local Langfuse initialization file |
| `make langfuse-up`, `make langfuse-down` | Start or stop the loopback Langfuse stack; `down` preserves volumes |
| `make dev` | `langgraph dev` on port 2024: graphs, webhooks, dashboard API, and the bundled dashboard when a build exists |
| `make dev-ui` | `make web` and `make dev` together, the backend fronting Vite so the UI hot-reloads on port 2024 |
| `make web` | The Vite dev server alone on port 3000 |
| `make build-dashboard` | Installs the dashboard's dependencies and builds it into `ui/.output/public` |
| `make tunnel NGROK_DOMAIN=…` | `ngrok http 2024` on your static domain, exposing only `/webhooks/*` |
| `make run` | The FastAPI app alone on port 8000, no LangGraph runtime |
| `make desktop` | The Electron app in development, against a backend on port 2024 |
| `make install` | `uv sync --extra dev` |
| `make test [TEST_FILE=tests/…]` | `pytest -vvv` on `tests/` or the given path |
| `make lint`, `make format`, `make format-check` | ruff check and format (`format` rewrites files) |
| `make typecheck` | `ty check agent tests` |

## Troubleshooting

### Webhook not receiving events

- The tunnel must be running (`make tunnel`) against port 2024, and the URL in GitHub or Slack must be your ngrok domain. Do not swap in a tunnel that forwards the whole port; see [Optional: GitHub and Slack webhooks](#optional-github-and-slack-webhooks). GitHub shows each delivery under the App's **Advanced** tab; ngrok's inspector at `http://localhost:4040` shows what arrived. With the webhooks-only policy, ngrok itself answers 404 for anything outside `/webhooks/*`, so test with `/webhooks/slack`, not `/ok`.
- Restart the backend after changing `.env`: `langgraph dev` reloads on code changes only, so a new `GITHUB_WEBHOOK_SECRET` or `SLACK_SIGNING_SECRET` is not picked up until then, and every delivery is rejected as `Invalid signature` in the meantime. Slack then needs **Retry** on its Request URL under **Event Subscriptions**.
- Webhook secrets are required: without `GITHUB_WEBHOOK_SECRET`, `SLACK_SIGNING_SECRET`, or `LINEAR_WEBHOOK_SECRET`, every request to that endpoint is rejected with 401.

### Dashboard login fails or won't stay logged in

- `redirect_uri is not associated with this application`: the App must list `http://localhost:2024/dashboard/api/auth/callback` (or the `:3000` one when you open Vite directly). Add it in the App's settings.
- Login redirects but the session does not stick: keep local URLs on `http://` so the cookie is `SameSite=Lax`.
- `DASHBOARD_BASE_URL not configured` on Sign in with Slack or LangSmith: the backend has neither a dashboard build nor `DASHBOARD_DEV_SERVER_URL`, so it does not know where the dashboard is. Run `make build-dashboard` or use `make dev-ui`.
- Admin pages 403: add your GitHub login or email to `CONFIGURED_ADMINS`.

### Dashboard shows the LangGraph JSON instead of the UI, or 404s at `/`

- There is no dashboard build: run `make build-dashboard`, or use `make dev-ui`.
- With Vite on port 3000, `curl -i http://localhost:3000/dashboard/api/me` should return the backend's `401`, not HTML; otherwise export `DASHBOARD_API_URL` before `make web`.

### `Port 3000 is already in use`

`make dev-ui` and `make web` refuse to start when another Vite is still running (Vite is configured with `strictPort`). Stop the old one or check `lsof -iTCP:3000 -sTCP:LISTEN`.

For sandbox, token-encryption, and "agent not responding" problems, see the installation guide's [Troubleshooting](INSTALLATION.md#troubleshooting).
