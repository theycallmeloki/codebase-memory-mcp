/*
 * llm_client.c — the outbound model call, shared. See llm_client.h for why this
 * is a unit of its own rather than a static in the enrichment pass.
 */
#include "pipeline/llm_client.h"

#include "foundation/compat.h" /* cbm_fileno, cbm_fopen */
#include "foundation/compat_fs.h"
#include "foundation/constants.h"
#include "foundation/log.h"
#include "foundation/mem_core.h"
#include "foundation/platform.h"
#include "foundation/sha256.h"
#include "foundation/subprocess.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h> /* fchmod: the curl config carrying the token is 0600 */

/* Longest prefix of `buf` that ends on a complete UTF-8 sequence, for a slice
 * of `n` bytes cut out of a longer file.
 *
 * A byte-count cap lands wherever it lands, and a dangling lead byte makes the
 * whole slice invalid UTF-8. yyjson will not serialize an invalid string, so a
 * caller that embeds the slice in a request would silently drop the whole file
 * rather than analyse it in part: measured on a 76 KiB markdown file whose cut
 * fell on the lead byte of a box-drawing glyph, the file got no prose at all
 * while every sibling did. */
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

char *cbm_llm_read_file(const char *path, size_t cap, size_t *out_total) {
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

bool cbm_llm_write_file(const char *path, const char *body, size_t len, mode_t mode) {
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

char *cbm_llm_post(const char *url, const char *key, const char *body, int timeout_ms) {
    if (!url || !url[0] || !body) {
        return NULL;
    }
    const char *cache = cbm_resolve_cache_dir();
    if (!cache || !cache[0]) {
        return NULL;
    }
    char dir[CBM_PATH_MAX];
    if (snprintf(dir, sizeof(dir), "%s/llm", cache) <= 0 || !cbm_mkdir_p(dir, 0755)) {
        return NULL;
    }

    /* Temp files are named by the hash of the request body. Two callers with
     * different bodies therefore never share a path, which is what makes this
     * safe to call concurrently from a worker pool. */
    char sha[CBM_SHA256_HEX_LEN + 1];
    cbm_sha256_hex(body, strlen(body), sha);
    char req[CBM_PATH_MAX];
    char resp[CBM_PATH_MAX];
    char err[CBM_PATH_MAX];
    (void)snprintf(req, sizeof(req), "%s/%s.req", dir, sha);
    (void)snprintf(resp, sizeof(resp), "%s/%s.resp", dir, sha);
    (void)snprintf(err, sizeof(err), "%s/%s.err", dir, sha);

    if (!cbm_llm_write_file(req, body, strlen(body), 0)) {
        return NULL;
    }

    size_t ulen = strlen(url);
    while (ulen > 0 && url[ulen - 1] == '/') {
        ulen--; /* tolerate a trailing slash on the configured base */
    }

    char timeout_s[CBM_SZ_16];
    (void)snprintf(timeout_s, sizeof(timeout_s), "%d", (timeout_ms + 999) / 1000);
    char data_arg[CBM_PATH_MAX + 1];
    (void)snprintf(data_arg, sizeof(data_arg), "@%s", req);

    /* The bearer token goes in a curl config file, NEVER on argv. /proc/<pid>/
     * cmdline is world-readable, so `-H "Authorization: Bearer sk-..."` hands
     * the key to every local user who runs `ps` for as long as the request
     * lives. The config is created 0600 and unlinked as soon as the call
     * returns, so the secret exists on disk only for the duration. */
    char cfg[CBM_PATH_MAX];
    (void)snprintf(cfg, sizeof(cfg), "%s/%s.curl", dir, sha);
    {
        char cfg_body[CBM_PATH_MAX * 2 + 512];
        int n = snprintf(cfg_body, sizeof(cfg_body),
                         "url = \"%.*s/chat/completions\"\n"
                         "request = \"POST\"\n"
                         "header = \"Content-Type: application/json\"\n",
                         (int)ulen, url);
        if (n > 0 && (size_t)n < sizeof(cfg_body) && key && key[0]) {
            n += snprintf(cfg_body + n, sizeof(cfg_body) - (size_t)n,
                          "header = \"Authorization: Bearer %s\"\n", key);
        }
        if (n > 0 && (size_t)n < sizeof(cfg_body)) {
            n += snprintf(cfg_body + n, sizeof(cfg_body) - (size_t)n,
                          "data-binary = \"%s\"\noutput = \"%s\"\n", data_arg, resp);
        }
        if (n <= 0 || (size_t)n >= sizeof(cfg_body) ||
            !cbm_llm_write_file(cfg, cfg_body, (size_t)n, 0600)) {
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
    opts.quiet_timeout_ms = timeout_ms + 5000;

    cbm_proc_result_t res;
    memset(&res, 0, sizeof(res));
    int run_rc = cbm_subprocess_run(&opts, &res);
    (void)remove(cfg); /* holds the credential: drop it before anything else */
    if (run_rc != 0 || res.outcome != CBM_PROC_CLEAN) {
        cbm_log_info("llm.http", "outcome", cbm_proc_outcome_str(res.outcome));
        return NULL;
    }
    return cbm_llm_read_file(resp, CBM_LLM_RESPONSE_CAP, NULL);
}
