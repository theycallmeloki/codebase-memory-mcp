/*
 * test_contract_link.c — guards for the cross-repository contract-linking pass.
 *
 * These run in stub mode: CBM_LLM_URL is unset, so no model is called and
 * nothing is spent (the same convention pass_llm_enrich documents for its own
 * stub mode). That still exercises the whole path up to and including the
 * summary record, which is where the pass previously died: its numeric fields
 * are handed to cbm_log as (key, value) string pairs, and an int passed
 * straight through is read back as a pointer. Any reintroduction of that
 * crashes the test process rather than failing an assertion, which is the
 * correct signal for a segfault.
 *
 * What these do NOT cover: the response-free and edge-identity paths, which
 * need a model answer. Those are exercised end to end against a live endpoint,
 * not here.
 */
#include "test_framework.h"
#include "test_helpers.h"

#include "foundation/compat.h"
#include "pipeline/pass_contract_link.h"
#include "store/store.h"

#include <sqlite3/sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char cache[256];
    char *saved_url;
    bool url_was_set;
} contract_link_fixture_t;

static bool contract_link_fixture_begin(contract_link_fixture_t *fixture) {
    memset(fixture, 0, sizeof(*fixture));

    /* Stub mode is the contract these tests run under: the pass must reach its
     * "no model configured" skip without a network. */
    const char *saved = getenv("CBM_LLM_URL");
    if (saved) {
        fixture->saved_url = strdup(saved);
        if (!fixture->saved_url) {
            return false;
        }
        fixture->url_was_set = true;
    }
    if (cbm_unsetenv("CBM_LLM_URL") != 0) {
        free(fixture->saved_url);
        return false;
    }

    snprintf(fixture->cache, sizeof(fixture->cache), "/tmp/cbm-contract-link-XXXXXX");
    return cbm_mkdtemp(fixture->cache) != NULL &&
           cbm_setenv("CBM_CACHE_DIR", fixture->cache, 1) == 0;
}

static void contract_link_fixture_end(contract_link_fixture_t *fixture) {
    if (fixture->url_was_set && fixture->saved_url) {
        (void)cbm_setenv("CBM_LLM_URL", fixture->saved_url, 1);
    } else {
        (void)cbm_unsetenv("CBM_LLM_URL");
    }
    if (fixture->cache[0]) {
        th_rmtree(fixture->cache);
    }
    free(fixture->saved_url);
    memset(fixture, 0, sizeof(*fixture));
}

static bool contract_link_project_path(const contract_link_fixture_t *fixture, const char *project,
                                       char *out, size_t out_size) {
    int written = snprintf(out, out_size, "%s/%s.db", fixture->cache, project);
    return written > 0 && (size_t)written < out_size;
}

/* Seed one File node carrying a contract. `qualified_name` is the identity the
 * pass uses to point the far side of an edge at this file, so it is spelled
 * explicitly rather than derived. */
static bool contract_link_seed_file(const contract_link_fixture_t *fixture, const char *project,
                                    const char *file_path, const char *qualified_name,
                                    const char *contract) {
    char path[512];
    if (!contract_link_project_path(fixture, project, path, sizeof(path))) {
        return false;
    }
    cbm_store_t *store = cbm_store_open_path(path);
    if (!store) {
        return false;
    }
    bool ok = cbm_store_upsert_project(store, project, fixture->cache) == CBM_STORE_OK;
    char props[512];
    snprintf(props, sizeof(props), "{\"llm_contract\":\"%s\"}", contract);
    cbm_node_t file = {.project = project,
                       .label = "File",
                       .name = file_path,
                       .qualified_name = qualified_name,
                       .file_path = file_path,
                       .properties_json = props};
    int64_t id = ok ? cbm_store_upsert_node(store, &file) : 0;
    ok = ok && id > 0;
    cbm_store_close(store);
    return ok;
}

static int contract_link_count_edges(const contract_link_fixture_t *fixture, const char *project) {
    char path[512];
    if (!contract_link_project_path(fixture, project, path, sizeof(path))) {
        return -1;
    }
    cbm_store_t *store = cbm_store_open_path_existing(path);
    if (!store) {
        return -1;
    }
    int count = 0;
    struct sqlite3 *db = cbm_store_get_db(store);
    sqlite3_stmt *stmt = NULL;
    if (db && sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM edges WHERE type = 'SHARES_CONTRACT'",
                                 -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            count = sqlite3_column_int(stmt, 0);
        }
    }
    sqlite3_finalize(stmt);
    cbm_store_close(store);
    return count;
}

/* Two projects, two distinct contracts, no model configured. Runs the whole
 * collection path and the summary record; writes nothing. */
static int test_contract_link_stub_mode_reports_and_writes_no_edges(void) {
    contract_link_fixture_t fixture;
    ASSERT_TRUE(contract_link_fixture_begin(&fixture));
    ASSERT_TRUE(contract_link_seed_file(&fixture, "alpha-host", "src/a.ts", "src/a.ts",
                                        "every id must be derived from the row, not the index"));
    ASSERT_TRUE(contract_link_seed_file(&fixture, "beta-host", "src/b.ts", "src/b.ts",
                                        "a cache key must never collapse two distinct rows"));

    const char *targets[] = {"beta-host"};
    cbm_contract_link_result_t result;
    int rc = cbm_contract_link_run("alpha-host", targets, 1, &result);

    int alpha_edges = contract_link_count_edges(&fixture, "alpha-host");
    int beta_edges = contract_link_count_edges(&fixture, "beta-host");
    contract_link_fixture_end(&fixture);

    ASSERT_EQ(rc, 0);
    ASSERT_EQ(result.projects, 2);
    ASSERT_EQ(result.files_with_contract, 2);
    ASSERT_EQ(result.distinct_contracts, 2);
    /* No model reachable: no grouping is guessed, so no edge is written. */
    ASSERT_FALSE(result.model_answered);
    ASSERT_EQ(result.edges, 0);
    ASSERT_EQ(result.skipped_count, 0);
    ASSERT_EQ(alpha_edges, 0);
    ASSERT_EQ(beta_edges, 0);
    cbm_contract_link_result_free(&result);
    PASS();
}

/* One distinct contract cannot form a cross-project grouping, so the call is
 * never made even when a model is configured. */
static int test_contract_link_single_contract_skips_without_calling(void) {
    contract_link_fixture_t fixture;
    ASSERT_TRUE(contract_link_fixture_begin(&fixture));
    const char *shared = "identifiers must be derived from data, never from position";
    ASSERT_TRUE(contract_link_seed_file(&fixture, "gamma-host", "src/g.ts", "src/g.ts", shared));
    ASSERT_TRUE(contract_link_seed_file(&fixture, "delta-host", "src/d.ts", "src/d.ts", shared));

    const char *targets[] = {"delta-host"};
    cbm_contract_link_result_t result;
    int rc = cbm_contract_link_run("gamma-host", targets, 1, &result);
    contract_link_fixture_end(&fixture);

    ASSERT_EQ(rc, 0);
    ASSERT_EQ(result.projects, 2);
    ASSERT_EQ(result.distinct_contracts, 1);
    ASSERT_FALSE(result.model_answered);
    ASSERT_EQ(result.edges, 0);
    cbm_contract_link_result_free(&result);
    PASS();
}

/* A file whose contract cannot be attributed to a named node cannot be pointed
 * at from another repository, so it is not a member at all. */
static int test_contract_link_unnamed_file_is_not_a_member(void) {
    contract_link_fixture_t fixture;
    ASSERT_TRUE(contract_link_fixture_begin(&fixture));
    ASSERT_TRUE(contract_link_seed_file(&fixture, "epsilon-host", "src/e.ts", "",
                                        "every key must be stable across restarts"));
    ASSERT_TRUE(contract_link_seed_file(&fixture, "zeta-host", "src/z.ts", "",
                                        "a key must not collide with another row"));

    const char *targets[] = {"zeta-host"};
    cbm_contract_link_result_t result;
    int rc = cbm_contract_link_run("epsilon-host", targets, 1, &result);
    contract_link_fixture_end(&fixture);

    ASSERT_EQ(rc, 0);
    ASSERT_EQ(result.files_with_contract, 0);
    ASSERT_EQ(result.distinct_contracts, 0);
    ASSERT_EQ(result.edges, 0);
    cbm_contract_link_result_free(&result);
    PASS();
}

/* A link needs two repositories; the source alone is a no-op, not an error. */
static int test_contract_link_single_project_is_a_no_op(void) {
    contract_link_fixture_t fixture;
    ASSERT_TRUE(contract_link_fixture_begin(&fixture));
    ASSERT_TRUE(contract_link_seed_file(&fixture, "solitary-host", "src/s.ts", "src/s.ts",
                                        "each row needs its own derived identifier"));

    const char *targets[] = {"solitary-host"};
    cbm_contract_link_result_t result;
    int rc = cbm_contract_link_run("solitary-host", targets, 1, &result);
    contract_link_fixture_end(&fixture);

    ASSERT_EQ(rc, 0);
    ASSERT_EQ(result.projects, 0);
    ASSERT_EQ(result.files_with_contract, 0);
    ASSERT_EQ(result.edges, 0);
    cbm_contract_link_result_free(&result);
    PASS();
}

/* A target that was never indexed is reported, not silently dropped. */
static int test_contract_link_unindexed_target_is_reported(void) {
    contract_link_fixture_t fixture;
    ASSERT_TRUE(contract_link_fixture_begin(&fixture));
    ASSERT_TRUE(contract_link_seed_file(&fixture, "eta-host", "src/t.ts", "src/t.ts",
                                        "a derived identifier must survive a reserialisation"));

    const char *targets[] = {"absent-host"};
    cbm_contract_link_result_t result;
    int rc = cbm_contract_link_run("eta-host", targets, 1, &result);
    contract_link_fixture_end(&fixture);

    ASSERT_EQ(rc, 0);
    ASSERT_EQ(result.projects, 1);
    ASSERT_EQ(result.skipped_count, 1);
    ASSERT_STR_EQ(result.skipped[0].project, "absent-host");
    ASSERT_EQ(result.edges, 0);
    cbm_contract_link_result_free(&result);
    PASS();
}

SUITE(contract_link) {
    RUN_TEST(contract_link_stub_mode_reports_and_writes_no_edges);
    RUN_TEST(contract_link_single_contract_skips_without_calling);
    RUN_TEST(contract_link_unnamed_file_is_not_a_member);
    RUN_TEST(contract_link_single_project_is_a_no_op);
    RUN_TEST(contract_link_unindexed_target_is_reported);
}
