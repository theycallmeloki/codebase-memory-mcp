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
 * cache keyed by (model + path + content) makes a re-index of unchanged content
 * free, so a full rebuild does not re-bill.
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
#include "graph_buffer/graph_buffer.h"

#include "yyjson/yyjson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h> /* fchmod: the curl config carrying the bearer token is 0600 */

/* Property keys. Prefixed so they cannot collide with the structural keys the
 * extractor writes ("extension", "last_modified", "complexity", ...). */
static const char *const ENRICH_KEYS[] = {"llm_purpose", "llm_summary", "llm_business_context"};
enum {
    ENRICH_KEY_PURPOSE = 0,
    ENRICH_FIELD_COUNT = 3,
    ENRICH_VALUE_CAP = 768, /* per field, as written into properties */
    ENRICH_RESPONSE_CAP = 4 * 1024 * 1024,
};

/* ── configuration ──────────────────────────────────────────────────────── */

typedef struct {
    bool on;      /* false => no model configured; stub prose, no network */
    bool dry_run; /* report the projected work, spend nothing */
    char url[CBM_PATH_MAX];
    char key[CBM_PATH_MAX];
    char model[128];
    int timeout_ms;
    long max_bytes;
    long max_files; /* ceiling on files attempted in one run; 0 => none */
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

    /* Report what the pass WOULD spend, then stop, so a large repository can
     * be sized before committing to the call volume. */
    enrich_env("CBM_LLM_DRY_RUN", "0", num, sizeof(num));
    c->dry_run = num[0] == '1';

    /* A key is NOT required: a local server usually wants none. The URL is what
     * decides whether the pass talks to anything at all. */
    c->on = c->url[0] != '\0';
}

/* ── helpers ────────────────────────────────────────────────────────────── */

/* Longest prefix of `buf` that ends on a complete UTF-8 sequence, for a slice
 * of `n` bytes cut out of a longer file.
 *
 * A byte-count cap lands wherever it lands, and a dangling lead byte makes the
 * whole slice invalid UTF-8. yyjson will not serialize an invalid string, so
 * the request is never assembled and the file is dropped WITHOUT prose instead
 * of being analysed in part — a silent hole in the index. Measured on a 76 KiB
 * markdown file whose cut fell on the lead byte of a box-drawing glyph: the
 * file got no prose at all while every sibling did. */
static size_t utf8_safe_prefix(const char *buf, size_t n) {
    size_t i = 0;
    size_t last = 0; /* end of the last sequence that completed in full */
    while (i < n) {
        unsigned char c = (unsigned char)buf[i];
        size_t need = 1;
        if (c >= 0x80) {
            if ((c & 0xE0) == 0xC0) {
                need = 2;
            } else if ((c & 0xF0) == 0xE0) {
                need = 3;
            } else if ((c & 0xF8) == 0xF0) {
                need = 4;
            } else {
                return last; /* continuation byte with no lead: malformed */
            }
        }
        if (i + need > n) {
            return last; /* the sequence runs past the cut */
        }
        i += need;
        last = i;
    }
    return last;
}

/* Read up to `cap` bytes. When `out_total` is non-NULL it receives the file's
 * real size, which is how the caller learns the content was CUT rather than
 * merely short. */
static char *enrich_read_file(const char *path, size_t cap, size_t *out_total) {
    FILE *f = cbm_fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        (void)fclose(f);
        return NULL;
    }
    long size = ftell(f);
    if (size <= 0) {
        (void)fclose(f);
        return NULL;
    }
    (void)fseek(f, 0, SEEK_SET);
    size_t take = (size_t)size > cap ? cap : (size_t)size;
    if (out_total) {
        *out_total = (size_t)size;
    }
    char *buf = (char *)cbm_alloc(CBM_MEM_CLASS_EXTRACT, take + 1);
    if (!buf) {
        (void)fclose(f);
        return NULL;
    }
    size_t n = fread(buf, 1, take, f);
    (void)fclose(f);
    if (n < (size_t)size) {
        n = utf8_safe_prefix(buf, n); /* the cap cut mid-sequence: back off */
    }
    buf[n] = '\0';
    return buf;
}

static bool enrich_write_file_mode(const char *path, const char *body, size_t len, mode_t mode) {
    FILE *f = cbm_fopen(path, "wb");
    if (!f) {
        return false;
    }
    /* Narrow the mode BEFORE writing: a credential file that is briefly 0644
     * with the token already in it has leaked for that window. */
    if (mode != 0 && fchmod(cbm_fileno(f), mode) != 0) {
        (void)fclose(f);
        (void)remove(path);
        return false;
    }
    size_t n = fwrite(body, 1, len, f);
    bool ok = n == len;
    ok = (fclose(f) == 0) && ok;
    return ok;
}

static bool enrich_write_file(const char *path, const char *body, size_t len) {
    return enrich_write_file_mode(path, body, len, 0);
}

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
            static const char *const FIELDS[ENRICH_FIELD_COUNT] = {"purpose", "summary",
                                                                   "business_context"};
            ok = true;
            for (int i = 0; i < ENRICH_FIELD_COUNT; i++) {
                yyjson_val *v = yyjson_obj_get(obj, FIELDS[i]);
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

/* POST `body` to <url>/chat/completions and return the raw response. Both the
 * request and the response are files rather than argv/stdout: the body is a
 * whole source file, and the supervisor's log tail is line-oriented. */
static char *enrich_http_post(const enrich_cfg_t *c, const char *body) {
    const char *cache = cbm_resolve_cache_dir();
    if (!cache || !cache[0]) {
        return NULL;
    }
    char dir[CBM_PATH_MAX];
    if (snprintf(dir, sizeof(dir), "%s/llm", cache) <= 0 || !cbm_mkdir_p(dir, 0755)) {
        return NULL;
    }

    char sha[CBM_SHA256_HEX_LEN + 1];
    cbm_sha256_hex(body, strlen(body), sha);
    char req[CBM_PATH_MAX];
    char resp[CBM_PATH_MAX];
    char err[CBM_PATH_MAX];
    (void)snprintf(req, sizeof(req), "%s/%s.req", dir, sha);
    (void)snprintf(resp, sizeof(resp), "%s/%s.resp", dir, sha);
    (void)snprintf(err, sizeof(err), "%s/%s.err", dir, sha);

    if (!enrich_write_file(req, body, strlen(body))) {
        return NULL;
    }

    size_t ulen = strlen(c->url);
    while (ulen > 0 && c->url[ulen - 1] == '/') {
        ulen--; /* tolerate a trailing slash on the configured base */
    }

    char timeout_s[CBM_SZ_16];
    (void)snprintf(timeout_s, sizeof(timeout_s), "%d", (c->timeout_ms + 999) / 1000);
    char data_arg[CBM_PATH_MAX + 1];
    (void)snprintf(data_arg, sizeof(data_arg), "@%s", req);

    /* The bearer token goes in a curl config file, NEVER on argv. /proc/<pid>/
     * cmdline is world-readable, so `-H "Authorization: Bearer sk-..."` hands
     * the key to every local user who runs `ps` for as long as the request
     * lives. The config is created 0600 and unlinked as soon as the call
     * returns, so the secret exists on disk only for the duration of the
     * request. */
    char cfg[CBM_PATH_MAX];
    (void)snprintf(cfg, sizeof(cfg), "%s/%s.curl", dir, sha);
    {
        char cfg_body[CBM_PATH_MAX * 2 + 512];
        int n = snprintf(cfg_body, sizeof(cfg_body),
                         "url = \"%.*s/chat/completions\"\n"
                         "request = \"POST\"\n"
                         "header = \"Content-Type: application/json\"\n",
                         (int)ulen, c->url);
        if (n > 0 && (size_t)n < sizeof(cfg_body) && c->key[0]) {
            n += snprintf(cfg_body + n, sizeof(cfg_body) - (size_t)n,
                          "header = \"Authorization: Bearer %s\"\n", c->key);
        }
        if (n > 0 && (size_t)n < sizeof(cfg_body)) {
            n += snprintf(cfg_body + n, sizeof(cfg_body) - (size_t)n,
                          "data-binary = \"%s\"\noutput = \"%s\"\n", data_arg, resp);
        }
        if (n <= 0 || (size_t)n >= sizeof(cfg_body) ||
            !enrich_write_file_mode(cfg, cfg_body, (size_t)n, 0600)) {
            return NULL;
        }
    }

    const char *argv_curl[] = {"curl", "-sS", "--fail-with-body", "--max-time", timeout_s, "-K",
                               cfg,    NULL};

    cbm_proc_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.bin = "curl";
    opts.argv = argv_curl;
    opts.log_file = err; /* curl's diagnostics; never the request body */
    opts.delete_log_on_exit = true;
    opts.quiet_timeout_ms = c->timeout_ms + 5000;

    cbm_proc_result_t res;
    memset(&res, 0, sizeof(res));
    int run_rc = cbm_subprocess_run(&opts, &res);
    (void)remove(cfg); /* holds the credential: drop it before anything else */
    if (run_rc != 0 || res.outcome != CBM_PROC_CLEAN) {
        cbm_log_info("llm_enrich.http", "outcome", cbm_proc_outcome_str(res.outcome));
        return NULL;
    }
    return enrich_read_file(resp, ENRICH_RESPONSE_CAP, NULL);
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

    yyjson_mut_val *messages = yyjson_mut_arr(doc);

    yyjson_mut_val *sys = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, sys, "role", "system");
    yyjson_mut_obj_add_strcpy(
        doc, sys, "content",
        "You analyse one source file at a time for a code-search index. Reply with a single JSON "
        "object and nothing else, with exactly these keys: "
        "\"purpose\" (one sentence: what this file is for), "
        "\"summary\" (one short paragraph: what it does and how it fits the wider codebase), "
        "\"business_context\" (one sentence: which part of the product or workflow it serves). "
        "Describe only what the file actually shows. Never invent behaviour, callers or intent "
        "that is not visible in the content.");
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
    /* Key over the three inputs that decide the answer. Length-prefixed so a
     * path that is a prefix of another cannot collide with a shifted content. */
    char *key = (char *)cbm_alloc(CBM_MEM_CLASS_EXTRACT,
                                  strlen(c->model) + strlen(path) + strlen(content) + 64);
    if (!key) {
        return NULL;
    }
    (void)snprintf(key, strlen(c->model) + strlen(path) + strlen(content) + 64, "%s\n%zu:%s\n%s",
                   c->model, strlen(path), path, content);
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
    char *content = enrich_read_file(abs, (size_t)c->max_bytes, &total);
    if (!content) {
        return -1; /* unreadable, empty or oversized: leave the node alone */
    }

    /* Authoritative, content-addressed cache path. */
    char *real_cache = enrich_cache_path(c, rel, content);
    char *response = NULL;
    if (real_cache) {
        char *hit = enrich_read_file(real_cache, ENRICH_RESPONSE_CAP, NULL);
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
        response = enrich_http_post(c, request);
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
                enrich_write_file(tmp, response, strlen(response))) {
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
        if (!yyjson_mut_obj_put(root, yyjson_mut_strcpy(md, ENRICH_KEYS[i]),
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
 * would cost more than the check saves. */
static bool enrich_already_done(const cbm_gbuf_node_t *node) {
    const char *props = node ? node->properties_json : NULL;
    return props && strstr(props, ENRICH_KEYS[ENRICH_KEY_PURPOSE]) != NULL;
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

    char fields[ENRICH_FIELD_COUNT][ENRICH_VALUE_CAP];
    int enriched = 0;
    int skipped = 0;
    int failed = 0;

    for (int i = 0; i < file_count; i++) {
        const cbm_gbuf_node_t *node = files[i];
        if (!node || !node->file_path) {
            continue;
        }
        if (enrich_already_done(node)) {
            skipped++;
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

    char n_enriched[CBM_SZ_16];
    char n_skipped[CBM_SZ_16];
    (void)snprintf(n_enriched, sizeof(n_enriched), "%d", enriched);
    (void)snprintf(n_skipped, sizeof(n_skipped), "%d", skipped);
    cbm_log_info("llm_enrich.done", "model", cfg.on ? cfg.model : "(stub)", "enriched", n_enriched,
                 "skipped", n_skipped);
    if (failed > 0) {
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
