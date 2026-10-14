/*
 * llm_client.h — the outbound model call, shared by every pass that makes one.
 *
 * WHY THIS IS ITS OWN UNIT. Two passes now talk to a model (per-file
 * enrichment, and cross-repository contract canonicalisation), and the part
 * worth sharing is not the JSON building — that differs per pass — but the
 * transport, because the transport is where the credential is handled. curl is
 * driven through a 0600 config file that is unlinked the moment the request
 * returns; the bearer token never touches argv, because /proc/<pid>/cmdline is
 * world-readable and `ps` would hand the key to every local user for the life
 * of the request. A second copy of that reasoning is a second chance to get it
 * wrong, so it lives here once.
 *
 * TRANSPORT IS curl OVER THE FOUNDATION SUBPROCESS SUPERVISOR, which is the
 * project's only outbound-network mechanism. Vendoring an HTTP/TLS client would
 * end the zero-dependency claim, and this way HTTPS, proxies and redirects are
 * curl's problem. The same code path reaches a local llama.cpp/Ollama on plain
 * HTTP and a hosted endpoint over TLS.
 *
 * This unit deliberately knows nothing about prompts, prompts are the caller's
 * concern; it takes a finished request body and returns the raw response.
 */
#ifndef CBM_LLM_CLIENT_H
#define CBM_LLM_CLIENT_H

#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>

/* Ceiling on a model response body read back from disk. Comfortably above any
 * completion a pass here asks for; it exists so a runaway server cannot make a
 * caller allocate without bound. */
#define CBM_LLM_RESPONSE_CAP (4 * 1024 * 1024)

/* POST `body` to <url>/chat/completions and return the raw response body.
 *
 * `key` may be NULL or empty: a local server usually wants no credential, and
 * the URL is what decides whether a call happens at all.
 *
 * Returns a heap buffer the caller frees with free(), or NULL on any failure
 * (no cache dir, unwritable temp, curl non-zero, timeout, supervision failure).
 * A NULL return is not fatal anywhere a caller uses it — a pass that cannot
 * reach the model degrades to doing nothing, never to corrupting the graph. */
char *cbm_llm_post(const char *url, const char *key, const char *body, int timeout_ms);

/* Write `len` bytes to `path`, narrowing the mode to `mode` before the first
 * byte lands. Returns false on any error, removing a partial file. A mode of 0
 * means "do not touch the mode". Exposed because a caller that writes its own
 * credential-adjacent file should not re-derive the ordering: widening after
 * the write leaves the secret exposed for that window. */
bool cbm_llm_write_file(const char *path, const char *body, size_t len, mode_t mode);

/* Read up to `cap` bytes of `path`. Returns a NUL-terminated heap buffer the
 * caller frees with free(), or NULL when the file is missing, empty, unreadable
 * or larger than `cap`. When `out_total` is non-NULL it receives the file's real
 * size, which is how a caller learns the content was CUT rather than short. */
char *cbm_llm_read_file(const char *path, size_t cap, size_t *out_total);

#endif /* CBM_LLM_CLIENT_H */
