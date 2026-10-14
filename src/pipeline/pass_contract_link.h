/*
 * pass_contract_link.h — link files across repositories that enforce the same
 * DATA INVARIANT.
 *
 * WHY THIS EXISTS. The protocol matcher (pass_cross_repo.c) links repositories
 * that talk to each other: an HTTP call in one is matched to the route that
 * serves it in another. That covers services, and covers nothing else. Two
 * libraries that never call each other can still be responsible for the same
 * invariant — derive a stable identity from an object; never let two logically
 * different items collide onto one key — and that relationship is invisible to
 * every existing mechanism:
 *
 *   - identifiers do not match: the same idea is `selectId` in one repo,
 *     `__correlationKey` in another, `rowId` in a third;
 *   - per-file prose does not name it either, because the enrichment prompt asks
 *     "what is this file for?" and answers in package-local terms. Measured on a
 *     four-repo sample, that prompt named a data-derived key in ZERO files;
 *   - there is no call edge, no import and no shared package to follow.
 *
 * Measured per-file gold ranks on the Plumbline cross-repo case: the anchor repo
 * retrieves at ranks 2-64 while the sibling repos sit at 74, 105, and absent. The
 * files are simply not reachable from the question.
 *
 * HOW IT LINKS. The `contract` field the enrichment pass now writes (llm_contract)
 * is the same invariant stated repository-neutrally, so the extraction side is
 * already done. What remains is MATCHING: asked for a canonical slug per file,
 * a model returns four different slugs for one concept (`normalized-identity-
 * keyed-collection`, `row-selection-state-consistency`, ...), so an LLM-chosen
 * key cannot be the node identity. Instead the distinct contract strings are
 * handed to the model once, in a single grouping call, and it returns which of
 * them enforce the same generalised rule. That produced
 * "data-derived unique identity key" across four repositories on the sample.
 *
 * WHY EDGES AND NOT SHARED NODES. Every table in the store carries a single
 * `project` column; no node belongs to two projects today. The protocol matcher
 * already established the pattern for a cross-repository relationship: write the
 * edge into BOTH project databases, so neither side owns it. Reusing that needs
 * no schema change and no CBM_INDEX_FORMAT_VERSION bump. A shared, searchable
 * Contract node would be more powerful and is the natural next step, but it is a
 * strictly larger change and should be earned by evidence that the edges move
 * retrieval at all.
 *
 * SHAPE. Cross-project, so it cannot live in pipeline.c: every pass there runs
 * inside ONE project's graph buffer, and this one has to read peers. It follows
 * pass_cross_repo.c exactly — open each project's .db at <cache>/<project>.db
 * directly and insert edges into each.
 *
 * DEGRADATION. A model that is unreachable, or that answers with something that
 * does not parse, produces NO edges and a count of zero. It never guesses a
 * grouping, because a wrongly-linked pair is worse than a missing one: it would
 * assert a relationship between two files that does not exist, and retrieval
 * would trust it.
 */
#ifndef CBM_PASS_CONTRACT_LINK_H
#define CBM_PASS_CONTRACT_LINK_H

#include <stdbool.h>

/* One entry per project the run skipped, with a static reason string. */
typedef struct {
    char *project;      /* heap-owned */
    const char *reason; /* static */
} cbm_contract_skip_t;

typedef struct {
    int projects;            /* peer projects opened */
    int files_with_contract; /* File nodes carrying llm_contract */
    int distinct_contracts;  /* distinct contract strings handed to the model */
    int groups;              /* groups the model returned */
    int edges;               /* SHARES_CONTRACT edges written (both directions) */
    bool model_answered;     /* the grouping call returned a usable answer */

    cbm_contract_skip_t *skipped; /* projects that could not be read */
    int skipped_count;
} cbm_contract_link_result_t;

/* Link every file in `targets` (plus `source_project`) whose llm_contract
 * groups with a file in a DIFFERENT project.
 *
 * Returns 0 on success, which includes the ordinary "nothing to do" outcomes:
 * no projects, no contracts, or fewer than two distinct contracts. A non-zero
 * return means the run could not be attempted at all (bad arguments).
 *
 * `out` is filled in every case and must not be NULL. The caller frees the
 * `skipped` array with cbm_contract_link_result_free. */
int cbm_contract_link_run(const char *source_project, const char *const *targets, int target_count,
                          cbm_contract_link_result_t *out);

void cbm_contract_link_result_free(cbm_contract_link_result_t *result);

#endif /* CBM_PASS_CONTRACT_LINK_H */
