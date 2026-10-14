/*
 * pass_contract_link.c — cross-repository linking by data invariant.
 * See pass_contract_link.h for the why; this file is the how.
 */
#include "pipeline/pass_contract_link.h"

#include "foundation/compat.h" /* cbm_strdup */
#include "foundation/constants.h"
#include "foundation/log.h"
#include "foundation/mem_core.h"
#include "foundation/platform.h" /* cbm_resolve_cache_dir, cbm_safe_getenv */
#include "foundation/str_util.h" /* cbm_validate_project_name */
#include "pipeline/llm_client.h"
#include "pipeline/worker_pool.h"
#include "store/store.h"

#include "yyjson/yyjson.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    CL_MAX_PROJECTS = 64,
    CL_CONTRACT_CAP = 640,  /* one invariant sentence */
    CL_PROMPT_CAP = 65536,  /* the numbered contract list sent to the model */
    CL_MAX_CONTRACTS = 400, /* ceiling on the list; grouping is quadratic in prose */
    CL_EDGE_PROPS_CAP = 1024,
    CL_QN_CAP = 1024, /* one node's qualified name */
    /* Collection capacities. These are starting points for on-demand growth, not
     * ceilings: anything that CAPS the corpus silently drops work, and the drop
     * is ordered by node id rather than by relevance. */
    CL_MEMBER_INITIAL_CAP = 4096,
    CL_CONTRACT_INITIAL_CAP = 4096,
};

/* The edge type this pass writes. Named here so the writer and any reader agree
 * on one spelling. */
static const char *const CL_EDGE_TYPE = "SHARES_CONTRACT";

/* One file that carries a contract.
 *
 * `qn` is the identity that travels BETWEEN stores; `node_id` is only valid in
 * the store the member was scanned from. A peer's id means nothing in another
 * project's database — the same integer is some unrelated node there — so an
 * edge is written by resolving the peer's qualified name inside the store
 * receiving it. That is the same rule pass_cross_repo.c follows for Route QNs. */
typedef struct {
    int64_t node_id;
    int project_idx; /* index into the project table */
    char contract[CL_CONTRACT_CAP];
    char qn[CL_QN_CAP];
} cl_member_t;

typedef struct {
    cbm_store_t *store[CL_MAX_PROJECTS];
    const char *name[CL_MAX_PROJECTS];
    int count;
} cl_projects_t;

/* ── paths and stores ───────────────────────────────────────────────────── */

static const char *cl_cache_dir(void) {
    const char *dir = cbm_resolve_cache_dir();
    return dir ? dir : cbm_tmpdir();
}

static bool cl_db_path(const char *project, char *buf, size_t bufsz) {
    if (!project || !cbm_validate_project_name(project)) {
        return false;
    }
    int written = snprintf(buf, bufsz, "%s/%s.db", cl_cache_dir(), project);
    return written > 0 && (size_t)written < bufsz;
}

/* Open a peer project's database read-write, provided it really is that
 * project's store. Mirrors pass_cross_repo.c: a stale or foreign file at the
 * expected path must not be written to. */
static cbm_store_t *cl_open_project(const char *project) {
    char path[CBM_PATH_MAX];
    if (!cl_db_path(project, path, sizeof(path))) {
        return NULL;
    }
    cbm_store_t *store = cbm_store_open_path_existing(path);
    if (!store) {
        return NULL;
    }
    cbm_project_t *projects = NULL;
    int count = 0;
    bool found = false;
    if (cbm_store_check_integrity(store) &&
        cbm_store_list_projects(store, &projects, &count) == CBM_STORE_OK) {
        for (int i = 0; i < count; i++) {
            if (projects[i].name && strcmp(projects[i].name, project) == 0) {
                found = true;
                break;
            }
        }
    }
    cbm_store_free_projects(projects, count);
    if (!found) {
        cbm_store_close(store);
        return NULL;
    }
    return store;
}

/* ── contract extraction ────────────────────────────────────────────────── */

/* Read properties_json's "llm_contract". Returns false when absent or empty.
 * Parsed rather than substring-searched so an escape inside the sentence cannot
 * silently truncate it. */
static bool cl_contract_of(const char *properties_json, char *out, size_t outsz) {
    if (!properties_json || !properties_json[0]) {
        return false;
    }
    yyjson_doc *doc = yyjson_read(properties_json, strlen(properties_json), 0);
    if (!doc) {
        return false;
    }
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *v = yyjson_is_obj(root) ? yyjson_obj_get(root, "llm_contract") : NULL;
    const char *s = yyjson_is_str(v) ? yyjson_get_str(v) : NULL;
    bool ok = s && s[0];
    if (ok) {
        (void)snprintf(out, outsz, "%s", s);
    }
    yyjson_doc_free(doc);
    return ok;
}

/* ── grouping machinery ─────────────────────────────────────────────────── */

/* The contract corpus does not fit one prompt: the reference case has ~5000
 * distinct contracts at ~256 chars each, so a single request would be ~1.3 MB
 * against a 64 KB prompt ceiling — and the old fixed 400-contract list silently
 * DROPPED 92% of the corpus, in node-id order, which biased the links that could
 * ever be produced toward the first-indexed repo-root files. Instead each
 * project PAIR is grouped separately, so a window always contains both sides of
 * a possible cross-repository link, and each window is chunked to fit the
 * prompt. Groups found in different windows are merged with a union-find, so
 * overlap is not lost. */
enum { CL_CHUNK_CONTRACTS = 400 };

/* Union-find over distinct contracts. Iterative path compression: the contract
 * corpus is large enough that recursion depth is not worth the risk. */
static int cl_dsu_find(int *parent, int i) {
    int root = i;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[i] != root) {
        int next = parent[i];
        parent[i] = root;
        i = next;
    }
    return root;
}

static void cl_dsu_union(int *parent, int a, int b) {
    int ra = cl_dsu_find(parent, a);
    int rb = cl_dsu_find(parent, b);
    if (ra == rb) {
        return;
    }
    /* Lower index wins so the representative is stable and reproducible. */
    if (ra < rb) {
        parent[rb] = ra;
    } else {
        parent[ra] = rb;
    }
}

/* Defined below with the rest of the request/parse pair; declared here because a
 * chunk worker calls both. */
static char *cl_build_request(const char *url, const char *model, const char *const *contracts,
                              int count, int timeout_ms);
static int cl_parse_groups(const char *response_json, int contract_count, int *group_of);

/* One grouping request over a chunk of distinct contracts. */
typedef struct {
    const char *url;
    const char *key;
    const char *model;
    int timeout_ms;
    const char *const *distinct; /* the global contract list */
    const int *globals;          /* global distinct indices in this chunk */
    int count;
    int *local_group; /* out: chunk-local group id per entry, -1 when unparsed */
    bool ok;          /* out: request and parse both succeeded */
    int groups;       /* out: distinct local group ids returned */
} cl_chunk_t;

/* Runs on a pool thread. Each chunk owns its output arrays, so no synchronisation
 * is needed beyond the join at the end. */
static void cl_chunk_run(int idx, void *ctx) {
    cl_chunk_t *chunks = (cl_chunk_t *)ctx;
    cl_chunk_t *c = &chunks[idx];
    c->ok = false;
    c->groups = 0;
    const char **batch =
        (const char **)cbm_alloc(CBM_MEM_CLASS_EXTRACT, sizeof(*batch) * (size_t)c->count);
    if (!batch) {
        return;
    }
    for (int i = 0; i < c->count; i++) {
        batch[i] = c->distinct[c->globals[i]];
    }
    char *body = cl_build_request(c->url, c->model, batch, c->count, c->timeout_ms);
    cbm_free(CBM_MEM_CLASS_EXTRACT, batch);
    if (!body) {
        return;
    }
    char *response = cbm_llm_post(c->url, c->key, body, c->timeout_ms);
    free(body); /* cl_build_request -> yyjson_mut_write, the C allocator */
    if (!response) {
        return;
    }
    int groups = cl_parse_groups(response, c->count, c->local_group);
    /* cbm_llm_post -> cbm_llm_read_file -> cbm_alloc, so it is NOT free()-able. */
    cbm_free(CBM_MEM_CLASS_EXTRACT, response);
    if (groups <= 0) {
        return;
    }
    c->ok = true;
    c->groups = groups;
}

/* ── the grouping call ──────────────────────────────────────────────────── */

/* Ask the model which contracts enforce the same generalised rule.
 *
 * One call for the whole corpus, not one per pair: the model has to see the
 * candidates together to draw a boundary between them, and a pairwise formulation
 * is quadratic in the number of files. The reply is a partition of the 1-based
 * indices, so a contract the model considers unique simply lands in a group of
 * one and produces no edges.
 *
 * `max_tokens` is deliberately generous. Measured against deepseek-v4-flash, the
 * reasoning trace alone ran to ~1,445 tokens and an earlier attempt at 6,000
 * total hit finish=length with an EMPTY content field, which parses as "no
 * groups" and would look exactly like "nothing in common". */
static char *cl_build_request(const char *url, const char *model, const char *const *contracts,
                              int count, int timeout_ms) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) {
        return NULL;
    }
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "model", model);
    yyjson_mut_obj_add_bool(doc, root, "stream", false);
    yyjson_mut_obj_add_real(doc, root, "temperature", 0.0);
    yyjson_mut_val *fmt = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, fmt, "type", "json_object");
    yyjson_mut_obj_add_val(doc, root, "response_format", fmt);
    yyjson_mut_obj_add_int(doc, root, "max_tokens", 32000);

    yyjson_mut_val *messages = yyjson_mut_arr(doc);
    yyjson_mut_val *sys = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, sys, "role", "system");
    yyjson_mut_obj_add_strcpy(
        doc, sys, "content",
        "You group software invariants by the HAZARD each one guards against — the concrete "
        "failure it makes impossible — and never by shared wording, shared vocabulary, or the "
        "domain the file lives in. Two invariants belong together when the same underlying "
        "hazard, stated in each file's own terms, is what they prevent; a reader asking about "
        "that hazard should be sent to both. Look past surface phrasing: the same hazard is "
        "usually described with entirely different nouns in different libraries. Say an "
        "invariant guards against a lookup or correlation key that is derived from only PART "
        "of what makes two cases distinct, so that two logically different things can collide "
        "onto one key — invariants preventing that failure belong together even when one is "
        "about a cache, another about row selection, and a third about entity storage. "
        "Conversely, do NOT group invariants merely because they use similar words: two rules "
        "that both mention 'resolving a relative path' share wording, not necessarily hazard. "
        "Reply with JSON only: {\"groups\": [[<1-based indices>], ...]}. Include every index "
        "exactly once; put an invariant that matches nothing else in a group of its own.");
    yyjson_mut_arr_add_val(messages, sys);

    /* The candidate list. Truncated prose would change the grouping, so a
     * contract too long to include is dropped from the list entirely rather
     * than silently shortened. */
    char *listing = (char *)cbm_alloc(CBM_MEM_CLASS_EXTRACT, CL_PROMPT_CAP);
    if (!listing) {
        yyjson_mut_doc_free(doc);
        return NULL;
    }
    size_t used = 0;
    for (int i = 0; i < count; i++) {
        int n = snprintf(listing + used, CL_PROMPT_CAP - used, "%d. %s\n", i + 1, contracts[i]);
        if (n <= 0 || (size_t)n >= CL_PROMPT_CAP - used) {
            break;
        }
        used += (size_t)n;
    }
    yyjson_mut_val *user = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, user, "role", "user");
    yyjson_mut_obj_add_strcpy(doc, user, "content", listing);
    yyjson_mut_arr_add_val(messages, user);
    cbm_free(CBM_MEM_CLASS_EXTRACT, listing);

    yyjson_mut_obj_add_val(doc, root, "messages", messages);
    char *out = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);

    /* `url`, `model` and `timeout_ms` are consumed by the caller's POST; they are
     * parameters here to keep the request and its destination described in one
     * place. Silence unused-parameter by naming them in the signature only. */
    (void)url;
    (void)timeout_ms;
    return out;
}

/* Parse {"groups": [[1,2],[3]]} and mark, for each 1-based index, the group it
 * belongs to. Returns the number of groups, or -1 when the reply is unusable.
 * Out-of-range indices are ignored rather than fatal: a model that hallucinates
 * "7" must not corrupt a real grouping. */
static int cl_parse_groups(const char *response_json, int contract_count, int *group_of) {
    for (int i = 0; i < contract_count; i++) {
        group_of[i] = -1;
    }
    yyjson_doc *doc = yyjson_read(response_json, strlen(response_json), 0);
    if (!doc) {
        return -1;
    }
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *choices = yyjson_is_obj(root) ? yyjson_obj_get(root, "choices") : NULL;
    yyjson_val *first = yyjson_is_arr(choices) ? yyjson_arr_get_first(choices) : NULL;
    yyjson_val *message = yyjson_is_obj(first) ? yyjson_obj_get(first, "message") : NULL;
    yyjson_val *content = yyjson_is_obj(message) ? yyjson_obj_get(message, "content") : NULL;
    const char *inner = yyjson_is_str(content) ? yyjson_get_str(content) : NULL;

    int groups = -1;
    if (inner) {
        const char *start = strchr(inner, '{');
        yyjson_doc *payload = start ? yyjson_read(start, strlen(start), 0) : NULL;
        yyjson_val *obj = payload ? yyjson_doc_get_root(payload) : NULL;
        yyjson_val *arr = yyjson_is_obj(obj) ? yyjson_obj_get(obj, "groups") : NULL;
        if (yyjson_is_arr(arr)) {
            groups = 0;
            size_t gi = 0;
            size_t gj = 0;
            yyjson_val *g = NULL;
            yyjson_arr_foreach(arr, gi, gj, g) {
                int members = 0;
                if (yyjson_is_arr(g)) {
                    size_t mi = 0;
                    size_t mj = 0;
                    yyjson_val *m = NULL;
                    yyjson_arr_foreach(g, mi, mj, m) {
                        if (!yyjson_is_int(m)) {
                            continue;
                        }
                        int idx = (int)yyjson_get_int(m) - 1; /* 1-based on the wire */
                        if (idx >= 0 && idx < contract_count) {
                            group_of[idx] = groups;
                            members++;
                        }
                    }
                }
                if (members > 0) {
                    groups++;
                }
            }
        }
        if (payload) {
            yyjson_doc_free(payload);
        }
    }
    if (doc) {
        yyjson_doc_free(doc);
    }
    return groups;
}

/* ── edge writing ───────────────────────────────────────────────────────── */

/* Insert one direction of a cross-repository edge.
 *
 * The edge lands in `project`'s database and its `from_id` is that project's own
 * node. `to_id` is the peer's id *in the peer's database*, which is NOT
 * meaningful here — the same integer is some unrelated node in this store. The
 * remote endpoint is therefore identified the way pass_cross_repo.c identifies
 * its remote endpoints: in the property bag. `target_project` says which
 * database to open and `target_qn` says what to look up in it, so a reader never
 * trusts the foreign id. Both sides carry the pointer, so neither repo owns the
 * relationship. */
static bool cl_insert_edge(cbm_store_t *store, const char *project, int64_t from_id, int64_t to_id,
                           const char *target_project, const char *target_qn,
                           const char *contract) {
    char props[CL_EDGE_PROPS_CAP];
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) {
        return false;
    }
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "contract", contract);
    yyjson_mut_obj_add_strcpy(doc, root, "match", "llm_grouping");
    yyjson_mut_obj_add_strcpy(doc, root, "target_project", target_project);
    yyjson_mut_obj_add_strcpy(doc, root, "target_qn", target_qn);
    size_t len = 0;
    char *written = yyjson_mut_write(doc, 0, &len);
    yyjson_mut_doc_free(doc);
    if (!written) {
        return false;
    }
    (void)snprintf(props, sizeof(props), "%s", written);
    free(written);

    cbm_edge_t edge = {
        .project = project,
        .source_id = from_id,
        .target_id = to_id,
        .type = CL_EDGE_TYPE,
        .properties_json = props,
    };
    return cbm_store_insert_edge(store, &edge) > 0;
}

/* ── run ────────────────────────────────────────────────────────────────── */

/* cbm_log's variadic pairs are (const char *key, const char *value) — the value
 * is read back as a pointer, so an int passed straight in becomes an address
 * the logger dereferences and the process dies inside cbm_log with SIGSEGV.
 * Every numeric field below goes through this. The buffer is the caller's, so
 * the pointer stays valid for the duration of the log call. */
static const char *cl_i2s(char *buf, size_t cap, int value) {
    (void)snprintf(buf, cap, "%d", value);
    return buf;
}

static void cl_skip_add(cbm_contract_link_result_t *out, const char *project, const char *reason) {
    cbm_contract_skip_t *grown = realloc(out->skipped, sizeof(*grown) * (size_t)(out->skipped_count + 1));
    if (!grown) {
        return;
    }
    out->skipped = grown;
    out->skipped[out->skipped_count].project = cbm_strdup(project);
    out->skipped[out->skipped_count].reason = reason;
    out->skipped_count++;
}

void cbm_contract_link_result_free(cbm_contract_link_result_t *result) {
    if (!result || !result->skipped) {
        return;
    }
    for (int i = 0; i < result->skipped_count; i++) {
        free(result->skipped[i].project);
    }
    free(result->skipped);
    result->skipped = NULL;
    result->skipped_count = 0;
}

int cbm_contract_link_run(const char *source_project, const char *const *targets, int target_count,
                          cbm_contract_link_result_t *out) {
    if (!out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (!source_project || !cbm_validate_project_name(source_project)) {
        return -1;
    }

    /* ── 1. the project set: the source plus each target, deduped ─────────── */
    cl_projects_t proj;
    memset(&proj, 0, sizeof(proj));
    proj.name[0] = source_project;
    proj.count = 1;
    for (int i = 0; i < target_count && proj.count < CL_MAX_PROJECTS; i++) {
        const char *t = targets ? targets[i] : NULL;
        if (!t || !t[0] || strcmp(t, "*") == 0 || !cbm_validate_project_name(t)) {
            continue;
        }
        bool dup = false;
        for (int j = 0; j < proj.count; j++) {
            if (strcmp(proj.name[j], t) == 0) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            proj.name[proj.count++] = t;
        }
    }
    if (proj.count < 2) {
        cbm_log_info("contract_link.skip", "reason", "fewer_than_two_projects");
        return 0;
    }

    /* ── 2. collect every file that carries a contract ────────────────────── */
    /* Grown on demand. A fixed ceiling here silently discarded half the corpus
     * (the reference case has 5013 files carrying a contract), and WHICH half
     * survived was decided by node-id order — i.e. by indexing order, which for
     * a repo means its root-level files first. That biased the links the pass
     * could ever produce toward repo-root prose and away from source. */
    cl_member_t *members = NULL;
    int member_cap = 0;
    int member_count = 0;

    for (int p = 0; p < proj.count; p++) {
        cbm_store_t *store = cl_open_project(proj.name[p]);
        if (!store) {
            cl_skip_add(out, proj.name[p], "unreadable_store");
            continue;
        }
        proj.store[p] = store;
        out->projects++;

        cbm_node_t *nodes = NULL;
        int node_count = 0;
        if (cbm_store_find_nodes_by_label(store, proj.name[p], "File", &nodes, &node_count) ==
            CBM_STORE_OK) {
            for (int i = 0; i < node_count; i++) {
                char contract[CL_CONTRACT_CAP];
                if (!cl_contract_of(nodes[i].properties_json, contract, sizeof(contract))) {
                    continue;
                }
                out->files_with_contract++;
                if (member_count == member_cap) {
                    int next_cap = member_cap ? member_cap * 2 : CL_MEMBER_INITIAL_CAP;
                    cl_member_t *grown = (cl_member_t *)cbm_realloc(
                        CBM_MEM_CLASS_EXTRACT, members, sizeof(*grown) * (size_t)next_cap);
                    if (!grown) {
                        continue; /* out of room: keep counting, stop storing */
                    }
                    members = grown;
                    member_cap = next_cap;
                }
                cl_member_t *m = &members[member_count++];
                m->node_id = nodes[i].id;
                m->project_idx = p;
                (void)snprintf(m->contract, sizeof(m->contract), "%s", contract);
                /* Without a stable name there is nothing to look the peer up by
                 * in the other store, so a nameless node cannot take part. */
                (void)snprintf(m->qn, sizeof(m->qn), "%s",
                               nodes[i].qualified_name ? nodes[i].qualified_name : "");
                if (!m->qn[0]) {
                    member_count--;
                    out->files_with_contract--;
                    continue;
                }
            }
            cbm_store_free_nodes(nodes, node_count);
        }
    }
    if (member_count == 0) {
        cbm_log_info("contract_link.skip", "reason", "no_contracts");
        goto done;
    }

    /* ── 3. the distinct contracts, plus which projects carry each ────────── */
    char **distinct = NULL;
    uint64_t *carriers = NULL; /* bit t set when project t carries this contract */
    int distinct_cap = 0;
    int *index_of =
        (int *)cbm_alloc(CBM_MEM_CLASS_EXTRACT, sizeof(*index_of) * (size_t)member_count);
    if (!index_of) {
        cbm_free(CBM_MEM_CLASS_EXTRACT, members);
        return -1;
    }
    int distinct_count = 0;
    bool oom = false;
    for (int i = 0; i < member_count && !oom; i++) {
        int found = -1;
        for (int d = 0; d < distinct_count; d++) {
            if (strcmp(distinct[d], members[i].contract) == 0) {
                found = d;
                break;
            }
        }
        if (found < 0) {
            if (distinct_count == distinct_cap) {
                int next_cap = distinct_cap ? distinct_cap * 2 : CL_CONTRACT_INITIAL_CAP;
                char **grown =
                    (char **)cbm_realloc(CBM_MEM_CLASS_EXTRACT, distinct,
                                         sizeof(*grown) * (size_t)next_cap);
                if (!grown) {
                    oom = true;
                    break;
                }
                distinct = grown;
                uint64_t *grown_mask =
                    (uint64_t *)cbm_realloc(CBM_MEM_CLASS_EXTRACT, carriers,
                                            sizeof(*grown_mask) * (size_t)next_cap);
                if (!grown_mask) {
                    oom = true;
                    break;
                }
                carriers = grown_mask;
                distinct_cap = next_cap;
            }
            found = distinct_count++;
            /* Points into members[], which is fully built by now and never
             * grown again — reallocating it would dangle these. */
            distinct[found] = members[i].contract;
            carriers[found] = 0;
        }
        index_of[i] = found;
        carriers[found] |= (uint64_t)1 << (unsigned)members[i].project_idx;
    }
    out->distinct_contracts = distinct_count;
    if (oom || distinct_count < 2) {
        if (!oom) {
            cbm_log_info("contract_link.skip", "reason", "one_distinct_contract");
        }
        cbm_free(CBM_MEM_CLASS_EXTRACT, carriers);
        cbm_free(CBM_MEM_CLASS_EXTRACT, distinct);
        cbm_free(CBM_MEM_CLASS_EXTRACT, index_of);
        goto done;
    }
    /* Union-find parent over distinct contracts; the edge loop compares roots. */
    int *group_of = (int *)cbm_alloc(CBM_MEM_CLASS_EXTRACT, sizeof(*group_of) * (size_t)distinct_count);
    if (!group_of) {
        cbm_free(CBM_MEM_CLASS_EXTRACT, carriers);
        cbm_free(CBM_MEM_CLASS_EXTRACT, distinct);
        cbm_free(CBM_MEM_CLASS_EXTRACT, index_of);
        goto done;
    }
    for (int d = 0; d < distinct_count; d++) {
        group_of[d] = d;
    }

    /* ── 4. group every project PAIR, in chunks, in parallel ──────────────── */
    {
        char url[CBM_PATH_MAX];
        char key[CBM_PATH_MAX];
        char model[128];
        (void)cbm_safe_getenv("CBM_LLM_URL", url, sizeof(url), "");
        (void)cbm_safe_getenv("CBM_LLM_API_KEY", key, sizeof(key), "");
        (void)cbm_safe_getenv("CBM_LLM_MODEL", model, sizeof(model), "deepseek-chat");
        char timeout_env[CBM_SZ_32];
        (void)snprintf(timeout_env, sizeof(timeout_env), "%d", 300000);
        char timeout_buf[CBM_SZ_32];
        (void)cbm_safe_getenv("CBM_LLM_TIMEOUT_MS", timeout_buf, sizeof(timeout_buf), timeout_env);
        int timeout_ms = atoi(timeout_buf);
        if (timeout_ms <= 0) {
            timeout_ms = 300000;
        }
        if (!url[0]) {
            /* Unreachable model: write nothing. A guessed grouping would assert
             * a relationship between two files that may not exist. */
            cbm_log_info("contract_link.skip", "reason", "no_llm_url");
        } else {
            /* Window contents: for pair (a,b), every contract either project
             * carries. Windows are built for a pair so both ends of a possible
             * cross-repository link are always visible together — a flat window
             * over the whole corpus cannot group a db contract with a query one,
             * which is the only link this pass exists to produce.
             *
             * Sizes are computed first and `sel` allocated once, because chunk
             * descriptors hold pointers INTO it and growing it afterwards would
             * dangle them. */
            size_t sel_need = 0;
            for (int a = 0; a < proj.count; a++) {
                for (int b = a + 1; b < proj.count; b++) {
                    uint64_t want = ((uint64_t)1 << (unsigned)a) | ((uint64_t)1 << (unsigned)b);
                    for (int d = 0; d < distinct_count; d++) {
                        if (carriers[d] & want) {
                            sel_need++;
                        }
                    }
                }
            }
            int chunk_count = 0;
            for (size_t off = 0; off < sel_need; off += CL_CHUNK_CONTRACTS) {
                chunk_count++;
            }
            int *sel = sel_need ? (int *)cbm_alloc(CBM_MEM_CLASS_EXTRACT,
                                                   sizeof(*sel) * sel_need) : NULL;
            cl_chunk_t *chunks =
                chunk_count ? (cl_chunk_t *)cbm_alloc(CBM_MEM_CLASS_EXTRACT,
                                                      sizeof(*chunks) * (size_t)chunk_count)
                            : NULL;
            if (chunk_count == 0 || !sel || !chunks) {
                cbm_free(CBM_MEM_CLASS_EXTRACT, sel);
                cbm_free(CBM_MEM_CLASS_EXTRACT, chunks);
                cbm_log_info("contract_link.skip", "reason", "no_cross_project_candidates");
            } else {
                int written = 0;
                int made = 0;
                bool alloc_ok = true;
                for (int a = 0; a < proj.count && alloc_ok; a++) {
                    for (int b = a + 1; b < proj.count && alloc_ok; b++) {
                        uint64_t want =
                            ((uint64_t)1 << (unsigned)a) | ((uint64_t)1 << (unsigned)b);
                        int start = written;
                        for (int d = 0; d < distinct_count; d++) {
                            if (carriers[d] & want) {
                                sel[written++] = d;
                            }
                        }
                        for (int off = start; off < written; off += CL_CHUNK_CONTRACTS) {
                            int n = written - off;
                            if (n > CL_CHUNK_CONTRACTS) {
                                n = CL_CHUNK_CONTRACTS;
                            }
                            int *local = (int *)cbm_alloc(CBM_MEM_CLASS_EXTRACT,
                                                          sizeof(*local) * (size_t)n);
                            if (!local) {
                                alloc_ok = false;
                                break;
                            }
                            chunks[made++] = (cl_chunk_t){
                                .url = url,
                                .key = key,
                                .model = model,
                                .timeout_ms = timeout_ms,
                                .distinct = (const char *const *)distinct,
                                .globals = &sel[off],
                                .count = n,
                                .local_group = local,
                                .ok = false,
                                .groups = 0,
                            };
                        }
                    }
                }
                if (!alloc_ok) {
                    for (int i = 0; i < made; i++) {
                        cbm_free(CBM_MEM_CLASS_EXTRACT, chunks[i].local_group);
                    }
                    cbm_free(CBM_MEM_CLASS_EXTRACT, sel);
                    cbm_free(CBM_MEM_CLASS_EXTRACT, chunks);
                    cbm_log_info("contract_link.skip", "reason", "out_of_memory");
                } else {
                    char conc_buf[CBM_SZ_32];
                    (void)cbm_safe_getenv("CBM_LLM_CONCURRENCY", conc_buf, sizeof(conc_buf), "8");
                    int workers = atoi(conc_buf);
                    if (workers <= 0) {
                        workers = 8;
                    }
                    cbm_parallel_for_opts_t opts;
                    memset(&opts, 0, sizeof(opts));
                    opts.max_workers = workers;
                    cbm_parallel_for(made, cl_chunk_run, chunks, opts);

                    int answered = 0;
                    for (int i = 0; i < made; i++) {
                        cl_chunk_t *c = &chunks[i];
                        if (c->ok) {
                            answered++;
                        }
                        /* Merge within one chunk: every entry sharing a returned
                         * group id becomes one group globally. */
                        for (int x = 0; x < c->count; x++) {
                            if (c->local_group[x] < 0) {
                                continue;
                            }
                            for (int y = x + 1; y < c->count; y++) {
                                if (c->local_group[y] == c->local_group[x]) {
                                    cl_dsu_union(group_of, c->globals[x], c->globals[y]);
                                }
                            }
                        }
                    }
                    out->model_answered = answered > 0;
                    /* A group of one links nothing, so report only merged groups. */
                    int merged = 0;
                    {
                        int *sizes = (int *)cbm_alloc(CBM_MEM_CLASS_EXTRACT,
                                                      sizeof(*sizes) * (size_t)distinct_count);
                        if (sizes) {
                            for (int d = 0; d < distinct_count; d++) {
                                sizes[d] = 0;
                            }
                            for (int d = 0; d < distinct_count; d++) {
                                sizes[cl_dsu_find(group_of, d)]++;
                            }
                            for (int d = 0; d < distinct_count; d++) {
                                if (group_of[d] == d && sizes[d] >= 2) {
                                    merged++;
                                }
                            }
                            cbm_free(CBM_MEM_CLASS_EXTRACT, sizes);
                        }
                    }
                    out->groups = merged;
                    {
                        char chunk_buf[CBM_SZ_32];
                        char answered_buf[CBM_SZ_32];
                        cbm_log_info("contract_link.stage", "step", "grouped", "chunks",
                                     cl_i2s(chunk_buf, sizeof(chunk_buf), made), "answered",
                                     cl_i2s(answered_buf, sizeof(answered_buf), answered));
                    }
                    for (int i = 0; i < made; i++) {
                        cbm_free(CBM_MEM_CLASS_EXTRACT, chunks[i].local_group);
                    }
                    cbm_free(CBM_MEM_CLASS_EXTRACT, sel);
                    cbm_free(CBM_MEM_CLASS_EXTRACT, chunks);
                }
            }
        }
    }

    /* ── 5. edges between members of the same group in DIFFERENT projects ─── */
    for (int a = 0; a < member_count; a++) {
        if (index_of[a] < 0) {
            continue;
        }
        for (int b = a + 1; b < member_count; b++) {
            if (index_of[b] < 0) {
                continue;
            }
            /* Union-find roots, not raw ids: the same group can be discovered by
             * several pair windows and must compare equal across them. */
            if (cl_dsu_find(group_of, index_of[a]) != cl_dsu_find(group_of, index_of[b])) {
                continue;
            }
            if (members[a].project_idx == members[b].project_idx) {
                continue; /* within one repo the graph already connects them */
            }
            cbm_store_t *sa = proj.store[members[a].project_idx];
            cbm_store_t *sb = proj.store[members[b].project_idx];
            if (!sa || !sb) {
                continue;
            }
            const char *const ca = members[a].contract;
            /* Bidirectional, exactly as the protocol matcher writes CROSS_*:
             * neither project owns the relationship, and a query from either
             * side has to find it. Each direction is written into the database
             * of the member it starts from, and names the far side by project
             * plus qualified name, because the far side's node id is only valid
             * in the far side's database. */
            if (cl_insert_edge(sa, proj.name[members[a].project_idx], members[a].node_id,
                               members[b].node_id, proj.name[members[b].project_idx],
                               members[b].qn, ca)) {
                out->edges++;
            }
            if (cl_insert_edge(sb, proj.name[members[b].project_idx], members[b].node_id,
                               members[a].node_id, proj.name[members[a].project_idx],
                               members[a].qn, ca)) {
                out->edges++;
            }
        }
    }

    /* Section 3's arrays are freed here on the normal path; the early OOM exit
     * above frees them itself and jumps straight to done, so nothing is freed
     * twice. */
    cbm_free(CBM_MEM_CLASS_EXTRACT, carriers);
    cbm_free(CBM_MEM_CLASS_EXTRACT, index_of);
    cbm_free(CBM_MEM_CLASS_EXTRACT, group_of);
    cbm_free(CBM_MEM_CLASS_EXTRACT, distinct);
done:
    cbm_free(CBM_MEM_CLASS_EXTRACT, members);
    for (int p = 0; p < proj.count; p++) {
        if (proj.store[p]) {
            cbm_store_close(proj.store[p]);
        }
    }
    {
        char projects_buf[CBM_SZ_32];
        char contracts_buf[CBM_SZ_32];
        char groups_buf[CBM_SZ_32];
        char edges_buf[CBM_SZ_32];
        cbm_log_info("contract_link.done", "projects",
                     cl_i2s(projects_buf, sizeof(projects_buf), out->projects), "contracts",
                     cl_i2s(contracts_buf, sizeof(contracts_buf), out->distinct_contracts),
                     "groups", cl_i2s(groups_buf, sizeof(groups_buf), out->groups), "edges",
                     cl_i2s(edges_buf, sizeof(edges_buf), out->edges));
    }
    return 0;
}
