#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "negentropy.h"
#include "negentropy/storage/BTreeLMDB.h"
#include "negentropy/storage/SliceAELMDB.h"

#include "rbsr_aelmdb_test_util.h"

namespace {

static inline void reconcile(negentropy::StorageBase& storeA,
                             negentropy::StorageBase& storeB,
                             std::vector<std::string>& have,
                             std::vector<std::string>& need) {
    auto neA = negentropy::Negentropy(storeA);
    auto neB = negentropy::Negentropy(storeB);

    std::string msg = neA.initiate();
    while (true) {
        std::string response = neB.reconcile(msg);
        auto next = neA.reconcile(response, have, need);
        if (!next) break;
        msg = *next;
    }
}

// Minimal slice wrapper: present [lo, hi) of an existing store.
struct IndexSlice final : negentropy::StorageBase {
    negentropy::StorageBase& base;
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    IndexSlice(negentropy::StorageBase& base_, std::uint64_t lo_, std::uint64_t hi_)
        : base(base_), lo(lo_), hi(std::max(lo_, hi_)) {}

    std::uint64_t size() override { return hi - lo; }

    const negentropy::Item& getItem(size_t i) override {
        if (i >= size()) throw negentropy::err("IndexSlice: bad index");
        return base.getItem(size_t(lo + std::uint64_t(i)));
    }

    void iterate(size_t begin, size_t end,
                 std::function<bool(const negentropy::Item&, size_t)> cb) override {
        if (begin > end || end > size()) throw negentropy::err("IndexSlice: bad range");
        base.iterate(size_t(lo + begin), size_t(lo + end),
                     [&](const negentropy::Item& it, size_t abs_idx) {
                         const size_t rel = abs_idx - size_t(lo);
                         return cb(it, rel);
                     });
    }

    size_t findLowerBound(size_t begin, size_t end, const negentropy::Bound& value) override {
        if (begin > end || end > size()) throw negentropy::err("IndexSlice: bad range");
        const size_t abs = base.findLowerBound(size_t(lo + begin), size_t(lo + end), value);
        return abs - size_t(lo);
    }

    negentropy::Fingerprint fingerprint(size_t begin, size_t end) override {
        if (begin > end || end > size()) throw negentropy::err("IndexSlice: bad range");
        return base.fingerprint(size_t(lo + begin), size_t(lo + end));
    }
};

static inline std::pair<std::uint64_t, std::uint64_t>
compute_slice_indices(negentropy::StorageBase& store, std::uint64_t begin_ts, std::uint64_t end_ts) {
    const std::uint64_t full = store.size();
    const negentropy::Bound b0(begin_ts, "");
    const negentropy::Bound b1(end_ts, "");
    const std::uint64_t lo = store.findLowerBound(0, full, b0);
    const std::uint64_t hi = store.findLowerBound(0, full, b1);
    return {lo, hi};
}

static void test_aelmdb_slice_vs_slice() {
    using namespace rbsr_test;

    auto A = PeerAELMDB::create_fresh("/tmp/testdbA_aelmdb", "events");
    auto B = PeerAELMDB::create_fresh("/tmp/testdbB_aelmdb", "events");

    // Slice boundaries: [1200, 1800)
    const auto begin_kb = make_boundary_key(1200);
    const auto end_kb   = make_boundary_key(1800);

    const std::string_view begin_sv(reinterpret_cast<const char*>(begin_kb.data()), begin_kb.size());
    const std::string_view end_sv(reinterpret_cast<const char*>(end_kb.data()), end_kb.size());

    // Populate both peers with mostly common in-slice items, then add a few diffs.
    {
        auto txnA = lmdb::txn::begin(A.env);
        auto txnB = lmdb::txn::begin(B.env);

        for (std::uint64_t i = 0; i < 100; ++i) {
            const std::uint64_t ts = 1200 + i * 3;
            const auto id = pack_id_le(i);
            insert_item(txnA, A.dbi, ts, id);
            insert_item(txnB, B.dbi, ts, id);
        }

        // Inside-slice diffs
        insert_item(txnA, A.dbi, 1600, pack_id_le(100));
        insert_item(txnA, A.dbi, 1610, pack_id_le(101));
        insert_item(txnB, B.dbi, 1620, pack_id_le(200));

        // Outside-slice noise (must NOT affect reconciliation)
        insert_item(txnA, A.dbi, 1100, pack_id_le(300));
        insert_item(txnB, B.dbi, 1900, pack_id_le(400));

        txnA.commit();
        txnB.commit();
    }

    std::vector<std::string> have, need;
    {
        auto rtxnA = lmdb::txn::begin(A.env, nullptr, MDB_RDONLY);
        auto rtxnB = lmdb::txn::begin(B.env, nullptr, MDB_RDONLY);

        // IMPORTANT (named DBIs): force a DB record refresh in this txn before SliceAELMDB
        // reads per-DB fields like md_hash_offset via mdb_get_hash_offset().
        // Aggregate calls (e.g. totals/window APIs) already do the necessary root search.
        (void)A.dbi.totals(rtxnA);
        (void)B.dbi.totals(rtxnB);

        negentropy::storage::SliceAELMDB storeA(
            rtxnA, A.dbi,
            std::optional<std::string_view>{begin_sv},
            std::optional<std::string_view>{end_sv});

        negentropy::storage::SliceAELMDB storeB(
            rtxnB, B.dbi,
            std::optional<std::string_view>{begin_sv},
            std::optional<std::string_view>{end_sv});

        require(storeA.size() == 102, "unexpected AELMDB storeA slice size");
        require(storeB.size() == 101, "unexpected AELMDB storeB slice size");

        reconcile(storeA, storeB, have, need);
    }

    const auto have_u = ids_to_sorted_u64(have);
    const auto need_u = ids_to_sorted_u64(need);

    require(have_u == std::vector<std::uint64_t>({100, 101}), "bad have set (AELMDB slice)");
    require(need_u == std::vector<std::uint64_t>({200}), "bad need set (AELMDB slice)");

    require(std::find(have_u.begin(), have_u.end(), 300) == have_u.end(), "outside-slice A-only leaked");
    require(std::find(need_u.begin(), need_u.end(), 400) == need_u.end(), "outside-slice B-only leaked");

    std::cout << "OK (AELMDB slice <-> AELMDB slice reconciliation passed)\n";
}

static void test_aelmdb_slice_vs_btreelmdb_slice() {
    using namespace rbsr_test;

    // Peer A: AELMDB layout (one LMDB key per item, aggregates enabled).
    auto A = PeerAELMDB::create_fresh("/tmp/compatdb_A_aelmdb", "events");

    // Peer B: BTreeLMDB layout (packed nodes in LMDB values).
    auto Benv = FreshEnv::create_fresh("/tmp/compatdb_B_btree");
    lmdb::dbi btree_dbi{0};
    {
        auto txn = lmdb::txn::begin(Benv.env);
        btree_dbi = negentropy::storage::BTreeLMDB::setupDB(txn, "btree");
        txn.commit();
    }

    // Slice boundaries: [1200, 1800)
    const auto begin_kb = make_boundary_key(1200);
    const auto end_kb   = make_boundary_key(1800);
    const std::string_view begin_sv(reinterpret_cast<const char*>(begin_kb.data()), begin_kb.size());
    const std::string_view end_sv(reinterpret_cast<const char*>(end_kb.data()), end_kb.size());

    // Populate BOTH peers with common slice items, then introduce a few slice-only differences.
    {
        auto txnA = lmdb::txn::begin(A.env);
        auto txnB = lmdb::txn::begin(Benv.env);

        negentropy::storage::BTreeLMDB btreeB(txnB, btree_dbi, /*treeId=*/1);

        // Common in-slice
        for (std::uint64_t i = 0; i < 100; ++i) {
            const std::uint64_t ts = 1200 + i * 3;
            const auto id = pack_id_le(i);

            insert_item(txnA, A.dbi, ts, id);

            const std::string_view idsv(reinterpret_cast<const char*>(id.data()), id.size());
            (void)btreeB.insert(ts, idsv);
        }

        // Inside-slice diffs
        insert_item(txnA, A.dbi, 1600, pack_id_le(100));
        insert_item(txnA, A.dbi, 1610, pack_id_le(101));

        {
            const auto id200 = pack_id_le(200);
            const std::string_view idsv(reinterpret_cast<const char*>(id200.data()), id200.size());
            (void)btreeB.insert(1620, idsv);
        }

        // Outside-slice noise (must NOT affect reconciliation)
        insert_item(txnA, A.dbi, 1100, pack_id_le(300));
        {
            const auto id400 = pack_id_le(400);
            const std::string_view idsv(reinterpret_cast<const char*>(id400.data()), id400.size());
            (void)btreeB.insert(1900, idsv);
        }

        btreeB.flush();
        txnA.commit();
        txnB.commit();
    }

    std::vector<std::string> have, need;
    {
        auto rtxnA = lmdb::txn::begin(A.env, nullptr, MDB_RDONLY);
        auto rtxnB = lmdb::txn::begin(Benv.env, nullptr, MDB_RDONLY);

        // Same note as above: refresh named DBI metadata in this txn.
        (void)A.dbi.totals(rtxnA);

        // A side is already a slice store.
        negentropy::storage::SliceAELMDB sliceA(
            rtxnA, A.dbi,
            std::optional<std::string_view>{begin_sv},
            std::optional<std::string_view>{end_sv});

        // BTreeLMDB has no intrinsic slice; wrap it in IndexSlice.
        negentropy::storage::BTreeLMDB fullB(rtxnB, btree_dbi, /*treeId=*/1);
        const auto [loB, hiB] = compute_slice_indices(fullB, /*begin_ts=*/1200, /*end_ts=*/1800);
        IndexSlice sliceB(fullB, loB, hiB);

        require(sliceA.size() == 102, "unexpected SliceAELMDB slice size");
        require(sliceB.size() == 101, "unexpected BTreeLMDB slice size");

        reconcile(sliceA, sliceB, have, need);
    }

    const auto have_u = ids_to_sorted_u64(have);
    const auto need_u = ids_to_sorted_u64(need);

    require(have_u == std::vector<std::uint64_t>({100, 101}), "bad have set (AELMDB vs BTreeLMDB)");
    require(need_u == std::vector<std::uint64_t>({200}), "bad need set (AELMDB vs BTreeLMDB)");

    // Make sure outside-slice items are not present.
    require(std::find(have_u.begin(), have_u.end(), 300) == have_u.end(), "outside-slice A-only leaked");
    require(std::find(need_u.begin(), need_u.end(), 400) == need_u.end(), "outside-slice B-only leaked");

    std::cout << "OK (AELMDB slice <-> BTreeLMDB slice reconciliation passed)\n";
}

}  // namespace

int main() {
    test_aelmdb_slice_vs_slice();
    test_aelmdb_slice_vs_btreelmdb_slice();
    return 0;
}
