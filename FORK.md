# Fork notes

This is a fork of [`DeusData/codebase-memory-mcp`](https://github.com/DeusData/codebase-memory-mcp),
maintained at `theycallmeloki/codebase-memory-mcp`. It is **not** upstream, and
issues about the LLM behaviour belong here rather than there.

MIT licensed, © 2025 DeusData — see `LICENSE`. The fork keeps the copyright
notice and `THIRD_PARTY.md` intact; the release archive ships both.

## What this fork adds

Exactly one feature: **LLM-written per-file prose, indexed so natural-language
queries work.**

- **At index time**, a pass asks a model once per file for three short fields
  (`llm_purpose`, `llm_summary`, `llm_business_context`) and writes them into the
  `File` node's existing `properties_json` blob. No new column, no new table, no
  schema change.
- **At query time, nothing changes.** `nodes_fts.body` is derived from
  properties, so the prose simply joins the BM25 index that identifier search
  already uses. There is no model call on the query path and no new tool. Query
  latency is upstream's.

Everything else is upstream's code.

## Why a fork rather than a PR

Upstream's whole pitch is fast, local and free. This makes indexing
network-bound and metered: measured **3.2 s/file**, so a 270-file repository
takes **12m43s and 270 model calls** cold. That is a different product, not a
flag, so it lives here.

## Configuration

All knobs are environment variables; no config file is consulted.

| Variable | Default | Meaning |
|---|---|---|
| `CBM_LLM_URL` | *(unset)* | Base URL, e.g. `https://api.deepseek.com/v1`. **Unset disables the pass entirely** — it writes deterministic stub prose and makes no network call, so the pipeline stays exercisable offline. |
| `CBM_LLM_API_KEY` | *(empty)* | Bearer token. Omit for a local server that wants none. |
| `CBM_LLM_MODEL` | `deepseek-chat` | Model name. |
| `CBM_LLM_TIMEOUT_MS` | `60000` | Per-request ceiling. |
| `CBM_LLM_MAX_BYTES` | `65536` | Content cap sent per file. A larger file is sent **cut**, with an explicit truncation notice in the prompt — never silently. |
| `CBM_LLM_MAX_FILES` | `1000` | Ceiling on files attempted in one run. Exceeding it enriches **nothing** and reports why. `0` removes the ceiling. |
| `CBM_LLM_DRY_RUN` | `0` | `1` reports what the run would spend, then makes no calls. |

The API key is never passed on a command line: it goes to `curl` through a
`0600` config file that is unlinked the moment the request returns, so it cannot
be read out of `/proc/<pid>/cmdline`.

### Making the credentials actually reach the pass

Setting these in your shell is **not enough**, and this bites people silently.

The MCP server is not started by your shell — it is started by whichever agent
hosts it (omp, Claude Code, Codex, OpenCode, Cursor, …). Nothing you export
interactively reaches it. And the server starts a **long-lived daemon** that
serves every later index request with *its own* environment, so exporting the
variables after a daemon is already running changes nothing at all — including
for a brand-new file.

When the daemon has no `CBM_LLM_URL`, indexing still succeeds but every file
gets placeholder prose (`stub purpose: <path> is a <py> source file`) instead of
a description, and that text goes into the BM25 body of every file it touches.
The index result says so (`llm_enrich.kind == "stub"`), but only if you look.

`scripts/cbm-wire-llm.sh` fixes this. It writes a small wrapper that sources
your credentials file and execs the real binary, then points each client's MCP
`command` at that wrapper. The secret therefore stays in **one** file
(`~/.cbm-llm.env` by default) rather than being copied into every client config,
which is what per-client `env` blocks would require.

```sh
curl -fsSL https://raw.githubusercontent.com/theycallmeloki/codebase-memory-mcp/main/scripts/cbm-wire-llm.sh \
  | bash -s -- --dry-run     # see what it would change
cbm-wire-llm                 # do it
```

Then **restart your agent sessions** — a running daemon keeps the environment it
was started with. To check which environment is actually live:

```sh
pid=$(pgrep -f cbm-daemon-internal | head -1)
tr '\0' '\n' < /proc/$pid/environ | grep CBM_LLM
```

`CBM_LLM_URL` should be there. If it is not, that daemon predates the wiring.

Re-run `cbm-wire-llm` after any `install` — the installer rewrites the client
configs back to the bare binary path, which silently un-wires this. It is
idempotent, it backs each config up once (`.bak-cbm-llm`) on first change, and
`cbm-wire-llm --revert` restores them. `update` is safe: it replaces the binary,
not the wrapper.

## Cost model

- One model call per file that lacks prose, so spend scales with repository size.
- Results are cached on disk, keyed by `(model, path, content)`. **Re-indexing
  unchanged content makes no API call** — a warm full index of the 270-file
  test repo takes 5.4 s, not 12 minutes.
- Per-call bytes are bounded by `CBM_LLM_MAX_BYTES`.
- Per-run call count is bounded by `CBM_LLM_MAX_FILES`.

Size a repository before paying for it:

```sh
CBM_LLM_DRY_RUN=1 CBM_LLM_URL=... CBM_LLM_API_KEY=... \
  codebase-memory-mcp cli index_repository '{"repo_path":"/path","mode":"full"}'
```

The result carries the projection:

```json
"llm_enrich": { "kind": "dry_run",
                "detail": "dry run: would enrich 270 file(s), up to 17694720 bytes on the wire. No model calls were made." }
```

Hitting the ceiling says so the same way, with `"kind": "over_budget"` and the
remedy in `detail`. Neither is a log line: a CLI run logs quietly by default,
and the indexing worker's log file is unlinked on a clean exit, so the notice
rides the result where it can actually be read.

## Staying in sync with upstream

```sh
git fetch upstream
git merge upstream/main
```

**The rule that keeps this cheap: the feature lives in
`src/pipeline/pass_llm_enrich.c`; core files carry hooks only.** Never
restructure core code for the feature.

The hooks, all of them:

| File | Hook |
|---|---|
| `Makefile.cbm` | lists the one new source file |
| `src/pipeline/pipeline_internal.h` | the pass prototype |
| `src/pipeline/pipeline.c` | pass registration, and the notice setter/getters |
| `src/pipeline/pipeline.h` | the notice accessor declarations |
| `src/pipeline/pipeline_incremental.c` | calls the pass on **both** incremental routes |
| `src/store/store.c` | adds the prose keys to `FTS_BODY_EXPR`, and sizes `FTS_SQL_BUF` for it |
| `src/mcp/mcp.c` | renders the `llm_enrich` notice in the index result |

As of this writing the entire contribution is 9 files, **+1063/−19**: 723 lines
of new pass, 173 of release workflow, and roughly 170 lines of hooks. The first
upstream merge (3547 commits) conflicted in exactly one hook — `store.c`,
because upstream added `FTS_QN_EXPR` while this fork had raised `FTS_SQL_BUF`.
That is the expected shape of a future conflict: small, local, resolved by
keeping both.

## Releases

`fork-release.yml` builds on a `v*` tag and publishes a GitHub Release
containing a tarball and a `.sha256` for **linux/amd64** and **linux/arm64**.
The job smoke-tests `--version` and `--help` before publishing, so a binary that
cannot start is never released.

`fork-main.yml` builds and smoke-tests every push to `main` (with lint as an
advisory job) so that `main` is always taggable.

## Known limitations

- Linux only (amd64, arm64). No macOS or Windows binaries.
- Over-budget refuses the whole pass; there is no partial mode.
- The guard bounds **file count**, not tokens or currency — there is no spend
  accounting, only a ceiling and a dry-run projection.
- Upstream tests do not cover the new pass; it is verified by hand (see the
  commit messages for the measurements).
