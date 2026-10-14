/*
 * pass_llm_enrich.c — per-file natural-language enrichment.
 *
 * WHAT THIS ADDS. The structural graph already knows what a file DEFINES
 * (functions, classes, imports, calls) and says nothing about what the file is
 * FOR. That gap is why a natural-language question ("where do we decide which
 * job runs first?") cannot be answered by identifier search: the file that
 * governs the behaviour often shares no vocabulary with the question.
 *
 * This pass asks a model, once per file, for three short prose fields:
 *
 *     llm_purpose           one line: what this file is for
 *     llm_summary           a paragraph: what it does and how it fits
 *     llm_business_context  one line: which part of the product it serves
 *
 * WHERE IT GOES. Into the File node's EXISTING properties_json blob, exactly
 * like pass_importance's "importance" key: no new column, no new table, no
 * schema change, so CBM_INDEX_FORMAT_VERSION does not move and an index written
 * by an older build stays readable (the pass simply re-runs and fills the gap).
 *
 * HOW IT BECOMES SEARCHABLE. nodes_fts.body is derived from properties itself
 * (store.c FTS_BODY_EXPR), so that expression includes these three keys and the
 * prose joins the BM25 index smart_search already ranks. The label must also be
 * reachable: search_graph's label exclusion must not name File, or the prose is
 * indexed and never returned.
 *
 * TRANSPORT: curl, over the foundation subprocess supervisor. That is the
 * project's ONLY outbound-network mechanism (there is no linked HTTP/TLS client
 * and vendoring one would end the zero-dependency claim), and it means HTTPS,
 * proxies and redirects are curl's problem, not ours. It also reaches a local
 * llama.cpp/Ollama on plain HTTP and a hosted endpoint over TLS with the same
 * code path.
 *
 * CONFIGURATION is environment, mirroring the existing pipeline-level gate
 * (CBM_SEMANTIC_ENABLED): the pipeline sits below the tier that owns the config
 * database, and every pipeline toggle in this codebase is an env var.
 *
 *     CBM_LLM_URL         base URL, e.g. https://api.deepseek.com/v1
 *                         (or http://127.0.0.1:17890/v1 for llama.cpp).
 *                         UNSET => the pass writes deterministic stub prose and
 *                         makes no network call at all.
 *     CBM_LLM_API_KEY     bearer token; omit for a local server that wants none
 *     CBM_LLM_MODEL       default "deepseek-chat"
 *     CBM_LLM_TIMEOUT_MS  per-request ceiling, default 60000
 *     CBM_LLM_MAX_BYTES   file-content cap sent per file, default 65536. A file
 *                         larger than the cap is sent CUT, with an explicit
 *                         truncation notice in the prompt, never silently.
 *     CBM_LLM_MAX_FILES   ceiling on files attempted in one run, default 1000.
 *                         A run that would exceed it spends NOTHING and logs
 *                         llm_enrich.skip with the projection and the remedy,
 *                         rather than enriching a prefix. 0 removes the ceiling.
 *     CBM_LLM_CONCURRENCY model calls in flight at once, default 8. The stage
 *                         that talks to the model runs in a worker pool; the
 *                         stage that writes prose into the graph stays on the
 *                         main thread, so no graph-buffer lock is involved.
 *                         Set 1 for the strictly serial path. This is a
 *                         throughput knob, not a spend knob: the same number of
 *                         calls is made either way.
 *     CBM_LLM_NO_THINKING 1 => send chat_template_kwargs {"enable_thinking":
 *                         false} on every request, for a thinking model served
 *                         by llama.cpp/Ollama/vLLM. Summarising a file needs no
 *                         reasoning trace, and the trace costs full-price
 *                         output tokens: measured on Ternary-Bonsai-2-27B, the
 *                         same 5,443-token prompt took 820 completion tokens
 *                         and 37.7s with thinking on, 325 and 11.4s with it
 *                         off. Default 0, because the field is not part of the
 *                         OpenAI schema and a hosted backend may reject it.
 *     CBM_LLM_DRY_RUN     1 => report how many files would be enriched and how
 *                         many bytes that could put on the wire, then make no
 *                         calls at all. For sizing a repository up front.
 *
 * The bearer token never appears on argv: /proc/<pid>/cmdline is world
 * readable, so it is passed to curl through a 0600 config file that is
 * unlinked the moment the request returns.
 *
 * COST. One call per file, and only on files the run actually visits — the
 * closure-delta route stages only changed files. A content-addressed on-disk
 * cache keyed by (model + prompt + path + content) makes a re-index of unchanged
 * content free, so a full rebuild does not re-bill. The PROMPT is part of that
 * key, not just the content: without it a prompt edit would silently reuse
 * answers the previous prompt produced, and since those answers still parse the
 * regression would be invisible. Editing the prompt therefore re-bills the
 * corpus once, which is the correct trade.
 */
#include "foundation/compat.h" /* cbm_fileno */
#include "foundation/constants.h"
#include "foundation/compat_fs.h"
#include "foundation/log.h"
#include "foundation/mem_core.h"
#include "foundation/platform.h"
#include "foundation/sha256.h"
#include "foundation/subprocess.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline_internal.h"
#include "pipeline/llm_client.h"
#include "pipeline/worker_pool.h"
#include "graph_buffer/graph_buffer.h"

#include "yyjson/yyjson.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h> /* fchmod: the curl config carrying the bearer token is 0600 */

/* The fields the model returns, paired with the property keys they are stored
 * under. Property keys are prefixed so they cannot collide with the structural
 * keys the extractor writes ("extension", "last_modified", "complexity", ...).
 * One table, so a new field cannot be added to one list and forgotten in the
 * other: the parser reads .response_key and the merge writes .property_key, and
 * both iterate this array. Previously the response names lived in a second array
 * inside the parser and had to be kept in step by hand. */
typedef struct {
    const char *response_key; /* key in the model's JSON reply */
    const char *property_key; /* key written into the node's properties_json */
} enrich_field_t;

static const enrich_field_t ENRICH_FIELDS[] = {
    {"purpose", "llm_purpose"},
    {"summary", "llm_summary"},
    {"business_context", "llm_business_context"},
    {"contract", "llm_contract"},
};
enum {
    ENRICH_KEY_PURPOSE = 0,
    ENRICH_FIELD_COUNT = (int)(sizeof(ENRICH_FIELDS) / sizeof(ENRICH_FIELDS[0])),
    ENRICH_VALUE_CAP = 768, /* per field, as written into properties */
};

/* The whole system prompt, in one place. It is part of the cache key (see
 * enrich_cache_path), so editing it invalidates every cached answer by
 * construction: the key materially cannot outlive the prompt that produced it.
 * A hand-maintained version number would be forgotten the first time someone
 * tweaks a sentence, and the failure is silent — the old answers still parse,
 * so a prompt change would quietly keep returning the previous prompt's
 * results. Hashing the text removes that whole class of mistake.
 *
 * The "contract" key is the cross-repository hook, and its wording is load
 * bearing. A prompt that asks only "what is this file for?" gets package-local
 * answers: on a four-repo sample it named a data-derived key in ZERO files, so
 * nothing could be linked. Asking for the DATA INVARIANT, in repository-neutral
 * terms and explicitly ignoring local identifiers, named one in ALL FOUR
 * ("the per-element identifier ... derived, never supplied by position" rather
 * than "Defines the entity adapter factory"). Measured, not assumed. */
static const char *const ENRICH_SYSTEM_PROMPT =
    "You analyse one source file at a time for a code-search index. Reply with a single JSON "
    "object and nothing else, with exactly these keys: "
    "\"purpose\" (one sentence: what this file is for), "
    "\"summary\" (one short paragraph: what it does and how it fits the wider codebase), "
    "\"business_context\" (one sentence: which part of the product or workflow it serves), "
    "\"contract\" (one sentence: the DATA INVARIANT this code is responsible for keeping true - "
    "what must hold about the data flowing through it such that violating it would be a bug. "
    "State it in repository-neutral terms and ignore local identifiers and package names; if "
    "another codebase implemented the same concept under a different name, your sentence must "
    "still apply to it). "
    "Describe only what the file actually shows. Never invent behaviour, callers or intent "
    "that is not visible in the content.";

/* ── configuration ──────────────────────────────────────────────────────── */

typedef struct {
    bool on;      /* false => no model configured; stub prose, no network */
    bool dry_run; /* report the projected work, spend nothing */
    char url[CBM_PATH_MAX];
    char key[CBM_PATH_MAX];
    char model[128];
    int timeout_ms;
    long max_bytes;
    long max_files;   /* ceiling on files attempted in one run; 0 => none */
    int concurrency;  /* model calls in flight at once; 1 => strictly serial */
    bool no_thinking; /* ask the server to skip its reasoning trace */
} enrich_cfg_t;

/* cbm_safe_getenv returns either the caller's buffer or the fallback, so the
 * result is always a valid string; copy from whatever came back. */
static void enrich_env(const char *name, const char *fallback, char *dst, size_t dstsz) {
    char tmp[CBM_PATH_MAX];
    const char *v = cbm_safe_getenv(name, tmp, sizeof(tmp), fallback);
    (void)snprintf(dst, dstsz, "%s", (v && v[0]) ? v : fallback);
}

static void enrich_cfg_load(enrich_cfg_t *c) {
    memset(c, 0, sizeof(*c));
    enrich_env("CBM_LLM_URL", "", c->url, sizeof(c->url));
    enrich_env("CBM_LLM_API_KEY", "", c->key, sizeof(c->key));
    enrich_env("CBM_LLM_MODEL", "deepseek-chat", c->model, sizeof(c->model));

    char num[CBM_SZ_32];
    enrich_env("CBM_LLM_TIMEOUT_MS", "60000", num, sizeof(num));
    int timeout = atoi(num);
    c->timeout_ms = timeout > 0 ? timeout : 60000;

    enrich_env("CBM_LLM_MAX_BYTES", "65536", num, sizeof(num));
    long cap = atol(num);
    c->max_bytes = cap > 0 ? cap : 65536;

    /* COST GUARD. This defaults to a CEILING rather than to unlimited, because
     * the pass spends one model call per file lacking prose and the repos big
     * enough for that to hurt are exactly the ones where nobody notices until
     * the invoice or the wall clock arrives. 0 disables the ceiling outright;
     * a run that hits it spends nothing and says so, with the remedy. */
    enrich_env("CBM_LLM_MAX_FILES", "1000", num, sizeof(num));
    long maxf = atol(num);
    c->max_files = maxf > 0 ? maxf : 0;

    /* THROUGHPUT. One model call per file, each dominated by a network round
     * trip (or, against a local llama.cpp, by decode). Serially that leaves a
     * 4-slot server idle between requests: measured against Ternary-Bonsai-2-
     * 27B on one RTX 3090, 8 files took 50.8s in flight but 8 x 8.9s = 71s one
     * at a time, and a 2,993-file repository is the difference between 40
     * minutes and 5.3 hours.
     *
     * Defaults to 8 rather than to core count: the ceiling that matters is
     * usually the server's own slot count, not the local CPU, and every worker
     * here is blocked on I/O. Set 1 to restore the strictly serial path. */
    enrich_env("CBM_LLM_CONCURRENCY", "8", num, sizeof(num));
    int conc = atoi(num);
    c->concurrency = conc > 0 ? conc : 1;

    /* REASONING TRACE. A thinking model spends output tokens on a chain of
     * thought before the answer, and those tokens are generated and billed one
     * at a time like any other. Summarising a file needs no reasoning trace.
     * Measured against Ternary-Bonsai-2-27B on a 5,443-token prompt: 820
     * completion tokens and 37.7s with thinking on; 325 tokens and 11.4s with
     * it off. Same model, same file, 3.3x.
     *
     * OFF BY DEFAULT. `chat_template_kwargs` is a llama.cpp/Ollama extension,
     * not part of the OpenAI schema, and a hosted backend may reject a body it
     * does not recognise. It is opt-in so the default request stays portable. */
    enrich_env("CBM_LLM_NO_THINKING", "0", num, sizeof(num));
    c->no_thinking = num[0] == '1';

    /* Report what the pass WOULD spend, then stop, so a large repository can
     * be sized before committing to the call volume. */
    enrich_env("CBM_LLM_DRY_RUN", "0", num, sizeof(num));
    c->dry_run = num[0] == '1';

    /* A key is NOT required: a local server usually wants none. The URL is what
     * decides whether the pass talks to anything at all. */
    c->on = c->url[0] != '\0';
}

/* ── helpers ────────────────────────────────────────────────────────────── */

/* File read/write and the model POST now live in pipeline/llm_client.{h,c}:
 * they are shared with the cross-repository contract pass, and the POST is
 * where the credential is handled, which is exactly the code that must not
 * exist in two places. */

/* ── the model call ─────────────────────────────────────────────────────── */

/* Copy a completed model response into the answer triple. Returns true only
 * when all three fields are present and non-empty. */
static bool enrich_parse_answer(const char *response_json,
                                char out[ENRICH_FIELD_COUNT][ENRICH_VALUE_CAP]) {
    bool ok = false;
    yyjson_doc *doc = yyjson_read(response_json, strlen(response_json), 0);
    yyjson_val *root = doc ? yyjson_doc_get_root(doc) : NULL;
    yyjson_val *choices = root ? yyjson_obj_get(root, "choices") : NULL;
    yyjson_val *first = yyjson_is_arr(choices) ? yyjson_arr_get_first(choices) : NULL;
    yyjson_val *message = first ? yyjson_obj_get(first, "message") : NULL;
    yyjson_val *content = message ? yyjson_obj_get(message, "content") : NULL;
    const char *inner = yyjson_is_str(content) ? yyjson_get_str(content) : NULL;
    if (inner) {
        /* The assistant turn is itself JSON: request it with response_format
         * json_object, but tolerate a fenced or chatty reply rather than
         * discarding a usable answer. */
        const char *start = strchr(inner, '{');
        yyjson_doc *payload = start ? yyjson_read(start, strlen(start), 0) : NULL;
        yyjson_val *obj = payload ? yyjson_doc_get_root(payload) : NULL;
        if (yyjson_is_obj(obj)) {
            ok = true;
            for (int i = 0; i < ENRICH_FIELD_COUNT; i++) {
                yyjson_val *v = yyjson_obj_get(obj, ENRICH_FIELDS[i].response_key);
                const char *s = yyjson_is_str(v) ? yyjson_get_str(v) : NULL;
                if (!s || !s[0]) {
                    ok = false;
                    break;
                }
                (void)snprintf(out[i], ENRICH_VALUE_CAP, "%s", s);
            }
        }
        if (payload) {
            yyjson_doc_free(payload);
        }
    }
    if (doc) {
        yyjson_doc_free(doc);
    }
    return ok;
}


/* Build the request body: a strict-JSON analyst prompt plus the file.
 *
 * `total` is the file's size on disk; `content` can be shorter when the file
 * hit CBM_LLM_MAX_BYTES. Truncation used to be silent, which is the worst
 * option: the system prompt says "describe only what the file actually shows",
 * and a model shown the first 64 KiB of a 116 KiB file cannot tell that it is
 * missing the rest, so it answers confidently about a file it only partly saw.
 * Say so in the prompt instead. */
static char *enrich_build_request(const enrich_cfg_t *c, const char *path, const char *content,
                                  size_t total) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) {
        return NULL;
    }
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_strcpy(doc, root, "model", c->model);
    yyjson_mut_obj_add_bool(doc, root, "stream", false);
    yyjson_mut_obj_add_real(doc, root, "temperature", 0.2);

    yyjson_mut_val *fmt = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, fmt, "type", "json_object");
    yyjson_mut_obj_add_val(doc, root, "response_format", fmt);

    /* Cheap insurance for the models that need it; see CBM_LLM_NO_THINKING. */
    if (c->no_thinking) {
        yyjson_mut_val *kwargs = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_bool(doc, kwargs, "enable_thinking", false);
        yyjson_mut_obj_add_val(doc, root, "chat_template_kwargs", kwargs);
    }

    yyjson_mut_val *messages = yyjson_mut_arr(doc);

    yyjson_mut_val *sys = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, sys, "role", "system");
    yyjson_mut_obj_add_strcpy(doc, sys, "content", ENRICH_SYSTEM_PROMPT);
    yyjson_mut_arr_add_val(messages, sys);

    yyjson_mut_val *user = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, user, "role", "user");
    size_t cap = strlen(path) + strlen(content) + 512;
    char *prompt = (char *)cbm_alloc(CBM_MEM_CLASS_EXTRACT, cap);
    if (!prompt) {
        yyjson_mut_doc_free(doc);
        return NULL;
    }
    (void)snprintf(prompt, cap, "File path: %s\n\nFile content:\n%s%s", path, content,
                   total > (size_t)c->max_bytes
                       ? "\n\n[TRUNCATED: the content above is only the beginning of this file. "
                         "Describe what you can see, and do not assert anything about the parts "
                         "you cannot.]"
                       : "");
    yyjson_mut_obj_add_strcpy(doc, user, "content", prompt);
    cbm_free(CBM_MEM_CLASS_EXTRACT, prompt);
    yyjson_mut_arr_add_val(messages, user);

    yyjson_mut_obj_add_val(doc, root, "messages", messages);

    char *json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    return json;
}

/* Read a cached answer; NULL when absent. The cache is content-addressed on
 * (model + path + content), so an unchanged file never re-bills. */
static char *enrich_cache_path(const enrich_cfg_t *c, const char *path, const char *content) {
    const char *cache = cbm_resolve_cache_dir();
    if (!cache || !cache[0]) {
        return NULL;
    }
    char dir[CBM_PATH_MAX];
    if (snprintf(dir, sizeof(dir), "%s/llm", cache) <= 0 || !cbm_mkdir_p(dir, 0755)) {
        return NULL;
    }
    /* Key over EVERY input that decides the answer: the model, the system
     * prompt, the path and the content. The prompt belongs here. Leaving it out
     * means a prompt edit silently reuses answers the previous prompt produced
     * — and because those answers still parse, nothing anywhere reports it. The
     * fix is to hash the prompt text rather than track a version number by
     * hand; the key then cannot outlive the prompt that produced it.
     * Length-prefixed so a path that is a prefix of another cannot collide with
     * a shifted content. */
    size_t cap =
        strlen(c->model) + strlen(ENRICH_SYSTEM_PROMPT) + strlen(path) + strlen(content) + 96;
    char *key = (char *)cbm_alloc(CBM_MEM_CLASS_EXTRACT, cap);
    if (!key) {
        return NULL;
    }
    (void)snprintf(key, cap, "%s\np%zu:%s\n%zu:%s\n%s", c->model, strlen(ENRICH_SYSTEM_PROMPT),
                   ENRICH_SYSTEM_PROMPT, strlen(path), path, content);
    char sha[CBM_SHA256_HEX_LEN + 1];
    cbm_sha256_hex(key, strlen(key), sha);
    cbm_free(CBM_MEM_CLASS_EXTRACT, key);

    char *out = (char *)cbm_alloc(CBM_MEM_CLASS_EXTRACT, CBM_PATH_MAX);
    if (!out) {
        return NULL;
    }
    (void)snprintf(out, CBM_PATH_MAX, "%s/%s.json", dir, sha);
    return out;
}

/* The model call. Returns 0 on success; out is filled with the three fields. */
static int enrich_call_model(const enrich_cfg_t *c, const char *repo_path,
                             const cbm_gbuf_node_t *node,
                             char out[ENRICH_FIELD_COUNT][ENRICH_VALUE_CAP]) {
    const char *rel = (node && node->file_path) ? node->file_path : NULL;
    if (!rel) {
        return -1;
    }

    if (!c->on) {
        /* No model configured: keep the pass exercised and deterministic.
         * The word "stub" is deliberate — it makes un-enriched prose obvious
         * in the graph, and gives the search check a token that cannot occur
         * by accident. */
        const char *dot = strrchr(rel, '.');
        const char *ext = (dot && dot[1]) ? dot + 1 : "unknown";
        (void)snprintf(out[0], ENRICH_VALUE_CAP, "stub purpose: %s is a %s source file", rel, ext);
        (void)snprintf(out[1], ENRICH_VALUE_CAP,
                       "stub summary: %s carries %s content; no model is configured "
                       "(CBM_LLM_URL unset), so this field is a placeholder.",
                       rel, ext);
        (void)snprintf(out[2], ENRICH_VALUE_CAP, "stub business context: not derived for %s", rel);
        return 0;
    }

    char abs[CBM_PATH_MAX];
    if (snprintf(abs, sizeof(abs), "%s/%s", repo_path ? repo_path : ".", rel) <= 0) {
        return -1;
    }

    size_t total = 0;
    char *content = cbm_llm_read_file(abs, (size_t)c->max_bytes, &total);
    if (!content) {
        return -1; /* unreadable, empty or oversized: leave the node alone */
    }

    /* Authoritative, content-addressed cache path. */
    char *real_cache = enrich_cache_path(c, rel, content);
    char *response = NULL;
    if (real_cache) {
        char *hit = cbm_llm_read_file(real_cache, CBM_LLM_RESPONSE_CAP, NULL);
        if (hit) {
            bool ok = enrich_parse_answer(hit, out);
            cbm_free(CBM_MEM_CLASS_EXTRACT, hit);
            if (ok) {
                cbm_free(CBM_MEM_CLASS_EXTRACT, real_cache);
                cbm_free(CBM_MEM_CLASS_EXTRACT, content);
                return 0;
            }
        }
    }

    char *request = enrich_build_request(c, rel, content, total);
    if (request) {
        response = cbm_llm_post(c->url, c->key, request, c->timeout_ms);
        /* yyjson_mut_write allocates with the C allocator, not cbm_alloc:
         * pairing it with cbm_free lands in mi_usable_size on a pointer the
         * tracked allocator does not own. Same rule as enrich_merge_props. */
        free(request);
    }
    cbm_free(CBM_MEM_CLASS_EXTRACT, content);

    bool ok = false;
    if (response) {
        ok = enrich_parse_answer(response, out);
        if (ok && real_cache) {
            /* Cache the raw response, not the parsed triple, so a parser change
             * can be re-run offline. Written via a rename: a truncated file
             * must never look like a cache hit. */
            char tmp[CBM_PATH_MAX];
            if (snprintf(tmp, sizeof(tmp), "%s.tmp", real_cache) > 0 &&
                cbm_llm_write_file(tmp, response, strlen(response), 0)) {
                (void)remove(real_cache);
                if (rename(tmp, real_cache) != 0) {
                    (void)remove(tmp);
                }
            }
        }
        cbm_free(CBM_MEM_CLASS_EXTRACT, response);
    }
    if (real_cache) {
        cbm_free(CBM_MEM_CLASS_EXTRACT, real_cache);
    }
    return ok ? 0 : -1;
}

/* ── JSON merge ─────────────────────────────────────────────────────────── */

/* A new properties string with the three keys set. Idempotent: yyjson_mut_obj_put
 * REPLACES an existing key rather than appending a duplicate, so re-running the
 * pass over a node that already has prose rewrites it in place. */
static char *enrich_merge_props(const char *old,
                                char fields[ENRICH_FIELD_COUNT][ENRICH_VALUE_CAP]) {
    const char *base = (old && old[0]) ? old : "{}";
    yyjson_doc *doc = yyjson_read(base, strlen(base), 0);
    if (!doc) {
        return NULL; /* unparseable properties: leave the node exactly as it was */
    }
    yyjson_mut_doc *md = yyjson_doc_mut_copy(doc, NULL);
    yyjson_doc_free(doc);
    if (!md) {
        return NULL;
    }
    yyjson_mut_val *root = yyjson_mut_doc_get_root(md);
    if (!root || !yyjson_mut_is_obj(root)) {
        yyjson_mut_doc_free(md);
        return NULL;
    }
    for (int i = 0; i < ENRICH_FIELD_COUNT; i++) {
        if (!yyjson_mut_obj_put(root, yyjson_mut_strcpy(md, ENRICH_FIELDS[i].property_key),
                                yyjson_mut_strcpy(md, fields[i]))) {
            yyjson_mut_doc_free(md);
            return NULL;
        }
    }
    char *out = yyjson_mut_write(md, 0, NULL);
    yyjson_mut_doc_free(md);
    return out; /* caller frees */
}

/* Does this node already carry prose? Used only to skip work when a buffer was
 * rebuilt for a file whose content did not change. A raw substring probe is
 * deliberate: properties are machine-written JSON, and a full parse per node
 * would cost more than the check saves.
 *
 * Placeholder prose needs no special case here. It would only matter if a
 * route ran the pass over a node whose properties still held a stub, and no
 * route does: every route that runs the pass recreates the File node first
 * (the closure route writes {"extension":...} and the merge replaces the row),
 * so the stub is gone before the check is reached. Verified against the
 * released binary: stubbing a repo and then re-indexing with the model
 * configured (one file changed, one added) replaced every placeholder with
 * real prose. Making placeholders "not done" would guard a case that cannot
 * occur. */
static bool enrich_already_done(const cbm_gbuf_node_t *node) {
    const char *props = node ? node->properties_json : NULL;
    return props && strstr(props, ENRICH_FIELDS[ENRICH_KEY_PURPOSE].property_key) != NULL;
}

/* ── parallel fetch ─────────────────────────────────────────────────────── */

/* One file's outcome from the concurrent fetch stage. The prose lands here
 * rather than in the node because writing it touches the graph buffer, and
 * that stays on the main thread. */
typedef struct {
    int status; /* 0 => fields are valid */
    char fields[ENRICH_FIELD_COUNT][ENRICH_VALUE_CAP];
} enrich_fetch_t;

typedef struct {
    const cbm_gbuf_node_t **nodes;
    int count;
    const enrich_cfg_t *cfg;
    const char *repo_path;
    enrich_fetch_t *out;
    atomic_int next;
} enrich_job_t;

/* Pull the next unclaimed file and call the model. Workers exit on the shared
 * counter rather than on a fixed partition, so one slow file cannot leave a
 * worker idle while another holds a queue. */
static void enrich_fetch_worker(int worker_id, void *arg) {
    (void)worker_id;
    enrich_job_t *job = (enrich_job_t *)arg;
    for (;;) {
        int i = atomic_fetch_add_explicit(&job->next, 1, memory_order_relaxed);
        if (i >= job->count) {
            break;
        }
        enrich_fetch_t *slot = &job->out[i];
        slot->status = enrich_call_model(job->cfg, job->repo_path, job->nodes[i], slot->fields);
    }
}

/* ── the pass ───────────────────────────────────────────────────────────── */

int cbm_pipeline_pass_llm_enrich(cbm_pipeline_ctx_t *ctx) {
    if (!ctx) {
        return 0;
    }
    cbm_gbuf_t *gb = ctx->gbuf;
    if (!gb) {
        return 0;
    }

    const cbm_gbuf_node_t **files = NULL;
    int file_count = 0;
    if (cbm_gbuf_find_by_label(gb, "File", &files, &file_count) != 0 || file_count == 0) {
        cbm_log_info("llm_enrich.skip", "reason", "no_files");
        return 0;
    }

    enrich_cfg_t cfg;
    enrich_cfg_load(&cfg);

    /* ── cost guard ───────────────────────────────────────────────────────
     * One model call per file that lacks prose, so the spend scales with the
     * size of the repository. Size the work BEFORE doing any of it: an
     * accidental run over a monorepo should report its own size rather than
     * quietly buy it.
     *
     * The count is an upper bound on calls, not a forecast. Files whose content
     * is already in the cache cost nothing, and this deliberately does not
     * consult the cache: discovering that would mean reading and hashing every
     * file up front, which is the work the guard exists to avoid. Erring high
     * is the safe direction for a ceiling.
     *
     * Stub mode (no CBM_LLM_URL) makes no call and spends nothing, so it is
     * not guarded. */
    if (cfg.on) {
        int need = 0;
        for (int i = 0; i < file_count; i++) {
            const cbm_gbuf_node_t *node = files[i];
            if (node && node->file_path && !enrich_already_done(node)) {
                need++;
            }
        }
        char n_need[CBM_SZ_16];
        (void)snprintf(n_need, sizeof(n_need), "%d", need);

        if (cfg.dry_run) {
            char n_bytes[CBM_SZ_32];
            char detail[CBM_SZ_256];
            (void)snprintf(n_bytes, sizeof(n_bytes), "%lld",
                           (long long)need * (long long)cfg.max_bytes);
            (void)snprintf(detail, sizeof(detail),
                           "dry run: would enrich %s file(s), up to %s bytes on the wire. "
                           "No model calls were made.",
                           n_need, n_bytes);
            /* The notice is what the operator sees; the log line is for a
             * profiling run, where the worker log survives. */
            cbm_pipeline_set_llm_notice(ctx->pipeline, "dry_run", detail);
            cbm_log_warn("llm_enrich.dry_run", "model", cfg.model, "would_enrich", n_need,
                         "worst_case_bytes", n_bytes);
            return 0;
        }
        if (cfg.max_files > 0 && (long)need > cfg.max_files) {
            /* Refuse wholesale rather than enriching a prefix: an index that is
             * half prose is harder to reason about than one with none, and the
             * remedy is the same either way. */
            char n_cap[CBM_SZ_16];
            char detail[CBM_SZ_256];
            (void)snprintf(n_cap, sizeof(n_cap), "%ld", cfg.max_files);
            (void)snprintf(detail, sizeof(detail),
                           "cost guard: %s file(s) need prose but CBM_LLM_MAX_FILES is %s, so "
                           "nothing was enriched and no model calls were made. Raise the "
                           "ceiling (0 disables it) and re-index.",
                           n_need, n_cap);
            cbm_pipeline_set_llm_notice(ctx->pipeline, "over_budget", detail);
            cbm_log_warn("llm_enrich.skip", "reason", "over_budget", "projected", n_need, "cap",
                         n_cap, "hint", "raise CBM_LLM_MAX_FILES, or 0 for no ceiling");
            return 0;
        }
    }

    int enriched = 0;
    int skipped = 0;
    int failed = 0;

    /* Stage 1 — claim the files that still need prose, before any thread
     * starts. Bounded by the node count, and the graph buffer is only read. */
    const cbm_gbuf_node_t **targets =
        cbm_alloc(CBM_MEM_CLASS_EXTRACT, (size_t)file_count * sizeof(*targets));
    enrich_fetch_t *fetched =
        targets ? cbm_calloc(CBM_MEM_CLASS_EXTRACT, (size_t)file_count * sizeof(*fetched)) : NULL;
    int target_count = 0;
    for (int i = 0; i < file_count; i++) {
        const cbm_gbuf_node_t *node = files[i];
        if (!node || !node->file_path) {
            continue;
        }
        if (enrich_already_done(node)) {
            skipped++;
            continue;
        }
        if (targets) {
            targets[target_count++] = node;
        }
    }

    /* Stage 2 — model calls, `concurrency` at a time. Stage 3 — merge, on the
     * main thread. The split is what keeps this safe: every file's temp paths
     * are keyed by sha(body) and the cache directory is thread-local, so the
     * workers never share a file, and cbm_gbuf_node_set_properties_json is
     * never called off the main thread, so the graph buffer needs no lock.
     *
     * Allocation failure degrades to the old strictly serial path over the
     * node list rather than aborting a run that is otherwise fine. */
    if (targets && fetched) {
        if (target_count > 0 && cfg.concurrency > 1) {
            enrich_job_t job = {
                .nodes = targets,
                .count = target_count,
                .cfg = &cfg,
                .repo_path = ctx->repo_path,
                .out = fetched,
            };
            atomic_init(&job.next, 0);
            cbm_parallel_for(cfg.concurrency, enrich_fetch_worker, &job,
                             (cbm_parallel_for_opts_t){.max_workers = cfg.concurrency,
                                                       .force_pthreads = false});
        } else {
            for (int i = 0; i < target_count; i++) {
                fetched[i].status =
                    enrich_call_model(&cfg, ctx->repo_path, targets[i], fetched[i].fields);
            }
        }
        for (int i = 0; i < target_count; i++) {
            if (fetched[i].status != 0) {
                failed++;
                continue;
            }
            char *merged = enrich_merge_props(targets[i]->properties_json, fetched[i].fields);
            if (!merged) {
                failed++;
                continue;
            }
            (void)cbm_gbuf_node_set_properties_json((cbm_gbuf_node_t *)targets[i], merged);
            free(merged);
            enriched++;
        }
    } else {
        char fields[ENRICH_FIELD_COUNT][ENRICH_VALUE_CAP];
        for (int i = 0; i < file_count; i++) {
            const cbm_gbuf_node_t *node = files[i];
            if (!node || !node->file_path || enrich_already_done(node)) {
                continue;
            }
            if (enrich_call_model(&cfg, ctx->repo_path, node, fields) != 0) {
                failed++;
                continue;
            }
            char *merged = enrich_merge_props(node->properties_json, fields);
            if (!merged) {
                failed++;
                continue;
            }
            (void)cbm_gbuf_node_set_properties_json((cbm_gbuf_node_t *)node, merged);
            free(merged);
            enriched++;
        }
    }
    if (fetched) {
        cbm_free(CBM_MEM_CLASS_EXTRACT, fetched);
    }
    if (targets) {
        cbm_free(CBM_MEM_CLASS_EXTRACT, targets);
    }

    char n_enriched[CBM_SZ_16];
    char n_skipped[CBM_SZ_16];
    (void)snprintf(n_enriched, sizeof(n_enriched), "%d", enriched);
    (void)snprintf(n_skipped, sizeof(n_skipped), "%d", skipped);
    cbm_log_info("llm_enrich.done", "model", cfg.on ? cfg.model : "(stub)", "enriched", n_enriched,
                 "skipped", n_skipped);
    if (!cfg.on && enriched > 0) {
        /* Placeholder prose is a misconfiguration, not a result. It looks like
         * a working index until someone reads it, and the placeholder text
         * itself ("stub", "purpose", "source file") pollutes the BM25 body of
         * every file it touches. Stub mode is only ever right for a deliberate
         * offline run, so it must not pass silently. */
        char detail[CBM_SZ_256];
        (void)snprintf(detail, sizeof(detail),
                       "%s file(s) got PLACEHOLDER prose: CBM_LLM_URL is unset, so no model was "
                       "called and the index now carries placeholder text instead of "
                       "descriptions. Set CBM_LLM_URL in the environment that starts the "
                       "daemon, then re-index.",
                       n_enriched);
        cbm_pipeline_set_llm_notice(ctx->pipeline, "stub", detail);
    } else if (failed > 0) {
        char n_failed[CBM_SZ_16];
        char n_attempted[CBM_SZ_16];
        char detail[CBM_SZ_256];
        (void)snprintf(n_failed, sizeof(n_failed), "%d", failed);
        (void)snprintf(n_attempted, sizeof(n_attempted), "%d", enriched + failed);
        /* Surfaced through the result for the same reason the guard is: this is
         * an INFO line, a CLI run suppresses INFO, and the worker's log file is
         * unlinked on a clean exit. Without it, a run that enriched 200 files
         * and failed on 70 is indistinguishable from one that enriched 200 of
         * 200 -- the index simply has less prose than it should, and nothing
         * anywhere says so. */
        (void)snprintf(detail, sizeof(detail),
                       "%s of %s file(s) could not be enriched; the rest of the index is "
                       "unaffected. Re-index to retry -- files that succeeded are cached and "
                       "cost nothing.",
                       n_failed, n_attempted);
        cbm_pipeline_set_llm_notice(ctx->pipeline, "partial", detail);
        cbm_log_info("llm_enrich.partial", "failed", n_failed);
    }
    return 0;
}
