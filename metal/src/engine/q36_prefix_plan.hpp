// q36_prefix_plan.hpp - the serve loop's prefix-cache decision, pure host logic (std-only, no Metal):
// match_len + plan_prefix + the checkpoint chain's bookkeeping.  Unit-tested by metal/tests/prefix_plan_test.cpp
// with a plain g++ build (the metal tree is macOS-only by CMake, so this header must not include mtl_rt.h).
//
// The state model behind the decision (q36_metal.cpp run(), serve loop): the KV cache is position-indexed, so
// rewinding s.pos is the whole KV rewind - attention reads only [0, pos) and the tail's re-evaluation overwrites
// the stale entries before anything reads them.  The GDN layers' S and conv_state are rolling: after a request
// they reflect the END of the live sequence, so a divergence at an earlier index needs the state as of that
// index - which exists only where the session saved a checkpoint.  The serve loop checkpoints each request's
// prompt end (depth 2) and restores the deepest checkpoint that still prefixes a diverging request.

#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace strata::qwen35 {

// One saved point of the session: the state as of having consumed exactly the first `pos` tokens, which are
// `tokens`.  `slot` tags which caller-side snapshot the state's buffers live in (the serve loop's two fixed
// S/conv/logits/hlast sets, Context-owned and reused in place): the header never interprets it, it only carries
// it so the serve loop knows which set to copy into and which set a restore copies from.
struct PrefixCheckpoint {
    size_t pos = 0;
    std::vector<int> tokens;
    int slot = 0;
};

struct PrefixPlan {
    size_t reuse = 0;        // the raw token-prefix match against the live sequence (bookkeeping)
    size_t eval_start = 0;   // where this request's evaluation resumes; what the PP lines and the DONE line's
                             // reused / actually-read fields report.  Tokens in [eval_start, reuse) match the
                             // live sequence but are re-evaluated: the checkpoint sits earlier, and evaluating
                             // the shared part again rebuilds exact state.
    int restore_idx = -1;    // the chain entry to restore from; -1: none
    bool full_reset = false; // the old T0 behavior: s.reset() + live_tokens cleared + the chain cleared
};

// The T0 match loop, extracted.
inline size_t match_len(const std::vector<int>& a, const std::vector<int>& b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[(size_t) n] == b[(size_t) n]) ++n;
    return n;
}

// The request-start decision.  A request that continues the live sequence (reuse == live.size(), a fresh
// session's first request included) evaluates only its delta from the live state.  A diverging request resumes
// from the deepest checkpoint whose tokens still prefix it; anything else (no checkpoint, none that prefixes,
// or one deeper than the request itself) is the old full reset.  Chain entries are only read through .pos and
// .tokens, so the serve loop can pass its own richer entry type.
template <typename Chain>
PrefixPlan plan_prefix(const std::vector<int>& live, const Chain& chain, const std::vector<int>& ids) {
    PrefixPlan plan;
    plan.reuse = match_len(live, ids);
    if (plan.reuse == live.size()) {
        plan.eval_start = plan.reuse;   // the delta starts at the live end
        return plan;
    }
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto& cp = chain[i];   // only .pos and .tokens are read: the serve loop may pass a richer type
        if (cp.tokens.empty()) continue;                  // an unfilled entry is not a checkpoint
        if (cp.pos > plan.reuse || cp.pos > ids.size()) continue;
        if (cp.tokens.size() != cp.pos) continue;         // a checkpoint's tokens are exactly its prefix
        // The chain invariant makes this redundant (every entry's tokens are a prefix of the live sequence,
        // and cp.pos <= reuse), but it is cheap and airtight - a divergence can leave an entry whose prefix
        // the next request no longer matches before the loop prunes it.
        if (!std::equal(cp.tokens.begin(), cp.tokens.end(), ids.begin())) continue;
        plan.restore_idx = (int) i;
        plan.eval_start = cp.pos;
    }
    plan.full_reset = plan.restore_idx < 0;
    return plan;
}

// Append this request's prompt end as the session's newest checkpoint; returns the snapshot slot the caller
// copies the state into (-1: the chain is disabled by cap 0).  A checkpoint at the current newest position
// replaces it in place (its slot and buffers are reused).  Past the cap the oldest entry leaves and its slot
// is recycled for the new entry - the caller's buffer sets persist per slot, so an erasure frees nothing.
inline int push_checkpoint(std::vector<PrefixCheckpoint>& chain, size_t cap, size_t pos, std::vector<int> tokens) {
    if (cap == 0) return -1;
    if (!chain.empty() && chain.back().pos == pos) {
        chain.back().tokens = std::move(tokens);
        return chain.back().slot;
    }
    if (chain.size() >= cap) chain.erase(chain.begin());
    // A slot no surviving entry holds.  After the erase above at most cap-1 entries survive, so a free slot
    // exists and the scan ends inside the range.
    int slot = 0;
    for (;;) {
        bool used = false;
        for (const auto& cp : chain)
            if (cp.slot == slot) { used = true; break; }
        if (!used) break;
        ++slot;
    }
    chain.push_back(PrefixCheckpoint{pos, std::move(tokens), slot});
    return slot;
}

// After a restore: entries deeper than the restored point leave.  Their S/conv copies are still a valid eval
// of their tokens, but the tail's re-evaluation overwrites the KV below their position - their state and the
// cache no longer agree, and their token prefix was cut by the divergence anyway.
inline void drop_deeper_than(std::vector<PrefixCheckpoint>& chain, size_t pos) {
    while (!chain.empty() && chain.back().pos > pos) chain.pop_back();
}

}   // namespace strata::qwen35
