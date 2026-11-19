// metal/tests/prefix_plan_test.cpp - the serve loop's prefix-cache decision as pure host logic (no Metal): the
// divergence restore's decision function and the checkpoint chain's bookkeeping (q36_prefix_plan.hpp).  This is
// T8-r1's sandbox gate - what the real engine does with the plan (the restore copies, the reuse counts) is the
// M2 session's check; this file pins the logic that decides it, including the cases the ladder never reaches.
//
//   cd metal/tests && g++ -std=c++20 -Wall -Wextra prefix_plan_test.cpp && ./a.out
//
// (The metal CMake glob also builds it on the Mac, where it links mtl_rt it never calls.)

#include "../src/engine/q36_prefix_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using strata::qwen35::PrefixCheckpoint;
using strata::qwen35::drop_deeper_than;
using strata::qwen35::plan_prefix;
using strata::qwen35::push_checkpoint;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

template <typename... Ts>
static std::vector<int> V(Ts... t) {
    return std::vector<int>{t...};
}

// The serve loop's entries carry more than the decision reads - the template must take them as they are.
struct Snap {
    size_t pos = 0;
    std::vector<int> tokens;
    int extra = 7;
};

int main() {
    std::printf("prefix_plan_test\n");

    std::printf(" extension (the T0 path, unchanged)\n");
    {
        auto chain = std::vector<PrefixCheckpoint>{PrefixCheckpoint{3, V(1, 2, 3), 0}};
        const auto plan = plan_prefix(V(1, 2, 3), chain, V(1, 2, 3, 4, 5));
        check(plan.reuse == 3 && plan.eval_start == 3 && plan.restore_idx == -1 && !plan.full_reset,
              "a request continuing live evaluates only its delta from the live state");
    }

    std::printf(" fresh session's first request\n");
    {
        const auto plan = plan_prefix({}, std::vector<PrefixCheckpoint>{}, V(9, 9, 9));
        check(plan.reuse == 0 && plan.eval_start == 0 && !plan.full_reset,
              "empty live extends from the zero state: evaluate all, no reset copy");
    }

    std::printf(" divergence at the checkpoint index (the ticket's scenario)\n");
    {
        auto chain = std::vector<PrefixCheckpoint>{PrefixCheckpoint{3, V(1, 2, 3), 0}};
        const auto plan = plan_prefix(V(1, 2, 3, 4, 5), chain, V(1, 2, 3, 7, 8));
        check(plan.reuse == 3 && plan.eval_start == 3 && plan.restore_idx == 0 && !plan.full_reset,
              "resume from the checkpoint; the divergent tail alone is read");
    }

    std::printf(" divergence deeper in the generated region\n");
    {
        auto chain = std::vector<PrefixCheckpoint>{PrefixCheckpoint{3, V(1, 2, 3), 1}};
        const auto plan = plan_prefix(V(1, 2, 3, 4, 5), chain, V(1, 2, 3, 4, 9));
        check(plan.reuse == 4 && plan.eval_start == 3 && plan.restore_idx == 0 && !plan.full_reset,
              "the shared part between the checkpoint and the divergence point is re-evaluated (4 != 3)");
    }

    std::printf(" divergence before every checkpoint\n");
    {
        auto chain = std::vector<PrefixCheckpoint>{PrefixCheckpoint{3, V(1, 2, 3), 0}};
        const auto plan = plan_prefix(V(1, 2, 3, 4, 5), chain, V(1, 2, 9, 9));
        check(plan.full_reset && plan.eval_start == 0 && plan.restore_idx == -1 && plan.reuse == 2,
              "no usable state that far back: the old full reset");
    }

    std::printf(" the depth-2 win: an edit between the two checkpoints\n");
    {
        auto chain = std::vector<PrefixCheckpoint>{PrefixCheckpoint{2, V(1, 2), 0}, PrefixCheckpoint{4, V(1, 2, 3, 4), 1}};
        const auto plan = plan_prefix(V(1, 2, 3, 4, 5, 6), chain, V(1, 2, 9, 9));
        check(plan.reuse == 2 && plan.eval_start == 2 && plan.restore_idx == 0 && !plan.full_reset,
              "the older checkpoint still prefixes the edited request: resume from it");
    }

    std::printf(" ids shorter than a checkpoint\n");
    {
        auto chain = std::vector<PrefixCheckpoint>{PrefixCheckpoint{4, V(1, 2, 3, 4), 0}};
        const auto plan = plan_prefix(V(1, 2, 3, 4, 5), chain, V(1, 2));
        check(plan.full_reset && plan.eval_start == 0,
              "a checkpoint deeper than the request itself cannot serve it");
        const auto plan2 = plan_prefix(V(1, 2, 3, 4, 5),
                                       std::vector<PrefixCheckpoint>{PrefixCheckpoint{2, V(1, 2), 0},
                                                                     PrefixCheckpoint{4, V(1, 2, 3, 4), 1}},
                                       V(1, 2, 9));
        check(!plan2.full_reset && plan2.eval_start == 2 && plan2.restore_idx == 0,
              "the shallower checkpoint still serves it");
    }

    std::printf(" exact re-send (the re-roll, empty tail)\n");
    {
        auto chain = std::vector<PrefixCheckpoint>{PrefixCheckpoint{3, V(1, 2, 3), 1}};
        const auto plan = plan_prefix(V(1, 2, 3, 4, 5), chain, V(1, 2, 3));
        check(plan.reuse == 3 && plan.eval_start == 3 && plan.restore_idx == 0 && !plan.full_reset,
              "ids.size() - eval_start == 0: sample straight from the checkpoint's logits");
    }

    std::printf(" the chain's bookkeeping\n");
    {
        std::vector<PrefixCheckpoint> chain;
        check(push_checkpoint(chain, 2, 2, V(1, 2)) == 0, "first push takes slot 0");
        check(push_checkpoint(chain, 2, 5, V(1, 2, 3, 4, 5)) == 1, "second push takes slot 1");
        check(push_checkpoint(chain, 2, 9, V(1, 2, 3, 4, 5, 6, 7, 8, 9)) == 0,
              "third push recycles the oldest entry's slot 0");
        check(chain.size() == 2 && chain[0].pos == 5 && chain[1].pos == 9,
              "cap 2 keeps the last two prompt ends, oldest first");
        check(push_checkpoint(chain, 2, 9, V(7, 7, 7, 7, 7, 7, 7, 7, 7)) == 0 && chain.size() == 2 &&
                  chain[1].pos == 9 && chain[1].tokens.size() == 9,
              "same-position push replaces the entry in place, same slot, no growth");
        drop_deeper_than(chain, 5);
        check(chain.size() == 1 && chain[0].pos == 5, "post-restore pruning drops the deeper entries");
        drop_deeper_than(chain, 0);
        check(chain.empty(), "pruning below every position empties the chain");
        drop_deeper_than(chain, 5);
        check(chain.empty(), "pruning an empty chain is a no-op");
    }

    std::printf(" the guard rails\n");
    {
        auto chain = std::vector<PrefixCheckpoint>{PrefixCheckpoint{2, V(1, 9), 0}};   // breaks the invariant
        const auto plan = plan_prefix(V(1, 2, 3), chain, V(1, 2, 7));
        check(plan.full_reset,
              "an entry whose tokens do not prefix the request is rejected, not restored");
        auto unfilled = std::vector<PrefixCheckpoint>{PrefixCheckpoint{}};
        const auto plan2 = plan_prefix(V(1, 2), unfilled, V(9, 9));
        check(plan2.full_reset,
              "an unfilled (empty-token) entry is never selected as a restore point");
        auto rich = std::vector<Snap>{Snap{3, V(1, 2, 3), 4}};
        const auto plan3 = plan_prefix(V(1, 2, 3, 4, 5), rich, V(1, 2, 3, 8));
        check(plan3.restore_idx == 0 && plan3.eval_start == 3,
              "the decision reads only .pos and .tokens: a richer serve-side entry type works");
        chain.clear();   // the caller's full-reset action
        const auto plan4 = plan_prefix({}, chain, V(4, 4));
        check(plan4.reuse == 0 && plan4.eval_start == 0 && !plan4.full_reset,
              "after a full reset the session starts over: next request extends from the zero state");
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "all green" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
