/*
  rbsr_aelmdb_advanced_test.cpp

  CSV column reference (one row per scenario/backend/outside-before-order):

  - scenario: Scenario name.
  - scenario_family/scenario_i/magnitude: grouping helpers (family+index derived from scenario name, magnitude from --magnitude).
  - backend: Storage backend used for the reconciliation (Vector | AELMDBSlice | BTreeLMDB).
  - fullA/fullB: Total items stored in A and B (entire DB / entire Vector).
  - sliceA/sliceB: Items within the [slice_begin, slice_end) boundaries for A and B.

  Scenario parameters (shape):
  - slice_begin_ts, slice_end_ts, step_in_slice
  - n_common_in_slice, n_a_only_in_slice, n_b_only_in_slice
  - n_common_outside_before/after
  - n_a_only_outside_before/after, n_b_only_outside_before/after

  Expected correctness denominators:
  - expected_have/expected_need: Expected counts based on persisted expected sets.

  Preparation (persisted in <root>/prep.meta, loaded by bench mode):
  - prep_open_ms: Time to open/create backing store used during preparation.
  - prep_populate_ms: Time to generate & insert all items into the store.
  - prep_commit_ms: LMDB-only: time spent committing transactions during preparation.
  - prep_seal_ms: Vector-only: time spent sealing vectors (sorting/materializing).
  - prep_expected_ms: Harness-only: time to compute + persist expected HAVE/NEED.
  - prep_serialize_ms: Vector-only: time to write A.bin/B.bin.

  Benchmark (measured during bench mode):
  - open_ms: Time to open the backing store (Vector: load A.bin/B.bin; LMDB: env/txn/dbi open).
  - build_ms: Time to build the slice wrapper (boundary/range plumbing).
  - reconcile_ms: Time for full Negentropy exchange + HAVE/NEED materialization.
  - decode_sort_ms: Portion of reconcile dedicated to unpacking ids + sorting.
  - total_bench_ms: open_ms + build_ms + reconcile_ms.

  Protocol stats:
  - msg_count: Total reconciliation messages exchanged.
  - bytes_a_to_b / bytes_b_to_a: Serialized protocol byte counts.
  - have / need: Counts of resulting item ids.

  Disk usage (bytes):
  - A_apparent_bytes/B_apparent_bytes: Sum of file sizes in peer dir (or A.bin/B.bin).
  - A_alloc_bytes/B_alloc_bytes: Allocated blocks (st_blocks*512 when available; else apparent).
  - A_used_bytes_est/B_used_bytes_est: LMDB estimate = (last_pgno+1)*page_size; Vector = bin file size.

  Memory:
  - rss_kb_before/rss_kb_after: RSS sampled around the bench run (Linux: /proc/self/status).

  Layout (LMDB-only, informational):
  - depth, branch_pages, leaf_pages, overflow_pages, last_pgno, page_size.

  Run config echoed:
  - outside_before_order: Order of insertion for outside-before items (asc|desc).
  - commit_every: LMDB preparation commit cadence.

  How to run:
    rm -rf /tmp/prepared_m5
    ./rbsr_aelmdb_advanced_test --mode both --magnitude 8 --mapsize-mb 2048 \
        --repeat-reconcile 10 --isolate-bench

*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
  #include <sys/stat.h>
  #include <sys/wait.h>
  #include <unistd.h>
#endif

#include "negentropy.h"
#include "negentropy/storage/Vector.h"
#include "negentropy/storage/BTreeLMDB.h"
#include "negentropy/storage/SliceAELMDB.h"
#include "rbsr_aelmdb_test_util_v2.h"


namespace rbsr_adv {

using rbsr_test::IdBytes;

struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
  Timer() : t0(clock::now()) {}
  double ms() const {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  }
};

static inline std::size_t read_rss_kb_linux() {
#if defined(__linux__)
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::size_t kb = 0;
      bool in_num = false;
      for (char c : line) {
        if (c >= '0' && c <= '9') { kb = kb * 10 + std::size_t(c - '0'); in_num = true; }
        else if (in_num) break;
      }
      return kb;
    }
  }
#endif
  return 0;
}


struct DiskBytes {
  std::uint64_t apparent = 0;
  std::uint64_t allocated = 0;
};

static inline std::uint64_t file_alloc_bytes(const std::filesystem::path& p) {
#if defined(__unix__) || defined(__APPLE__)
  struct stat st{};
  if (::stat(p.c_str(), &st) == 0) {
    return std::uint64_t(st.st_blocks) * 512ull;
  }
#endif
  std::error_code ec;
  const auto sz = std::filesystem::file_size(p, ec);
  return ec ? 0ull : std::uint64_t(sz);
}

static inline DiskBytes disk_bytes_of(const std::filesystem::path& p) {
  DiskBytes out;
  std::error_code ec;
  if (!std::filesystem::exists(p, ec) || ec) return out;

  if (std::filesystem::is_regular_file(p, ec) && !ec) {
    const auto sz = std::filesystem::file_size(p, ec);
    out.apparent = ec ? 0ull : std::uint64_t(sz);
    out.allocated = file_alloc_bytes(p);
    if (out.allocated == 0) out.allocated = out.apparent;
    return out;
  }

  if (!std::filesystem::is_directory(p, ec) || ec) return out;

  for (auto it = std::filesystem::recursive_directory_iterator(p, ec);
       !ec && it != std::filesystem::recursive_directory_iterator();
       it.increment(ec)) {
    if (ec) break;
    if (!it->is_regular_file(ec) || ec) { ec.clear(); continue; }

    const auto f = it->path();
    const auto sz = std::filesystem::file_size(f, ec);
    if (ec) { ec.clear(); continue; }
    out.apparent += std::uint64_t(sz);
    out.allocated += file_alloc_bytes(f);
  }
  if (out.allocated == 0) out.allocated = out.apparent;
  return out;
}

enum class Backend { AELMDBSlice, Vector, BTreeLMDB };

enum class OutsideBeforeOrder { Desc, Asc };

enum class Mode { InitOnly, BenchOnly, Both };

static inline const char* obefore_order_name(OutsideBeforeOrder o) {
  switch (o) {
    case OutsideBeforeOrder::Desc: return "desc";
    case OutsideBeforeOrder::Asc:  return "asc";
  }
  return "?";
}

static inline const char* backend_name(Backend b) {
  switch (b) {
    case Backend::AELMDBSlice: return "AELMDBSlice";
    case Backend::Vector:      return "Vector";
    case Backend::BTreeLMDB:   return "BTreeLMDB";
  }
  return "?";
}

// Generic index-slice wrapper for any StorageBase.
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

struct Scenario {
  std::string name;

  std::uint64_t slice_begin_ts = 0;
  std::uint64_t slice_end_ts = 0;

  std::uint64_t step_in_slice = 1;

  std::uint64_t n_common_in_slice = 0;
  std::uint64_t n_a_only_in_slice = 0;
  std::uint64_t n_b_only_in_slice = 0;

  std::uint64_t n_common_outside_before = 0;
  std::uint64_t n_common_outside_after  = 0;
  std::uint64_t n_a_only_outside_before = 0;
  std::uint64_t n_a_only_outside_after  = 0;
  std::uint64_t n_b_only_outside_before = 0;
  std::uint64_t n_b_only_outside_after  = 0;

  std::uint64_t base_common = 0;
  std::uint64_t base_a_only = 1'000'000;
  std::uint64_t base_b_only = 2'000'000;
  std::uint64_t base_out_common = 10'000'000;
  std::uint64_t base_out_a = 11'000'000;
  std::uint64_t base_out_b = 12'000'000;
};

struct DbLayout {
  std::uint64_t entries = 0;
  std::uint64_t depth = 0;
  std::uint64_t branch_pages = 0;
  std::uint64_t leaf_pages = 0;
  std::uint64_t overflow_pages = 0;
  std::uint64_t last_pgno = 0;  // env-level
  std::uint64_t page_size = 0;  // env-level (MDB_env_stat().ms_psize)
};

struct Metrics {
  // sizes
  std::uint64_t fullA = 0, fullB = 0;
  std::uint64_t sliceA = 0, sliceB = 0;

  // run config echoed in CSV
  OutsideBeforeOrder obefore_order = OutsideBeforeOrder::Desc;
  std::uint64_t commit_every = 0;     // preparation parameter
  std::uint64_t mapsize_mb = 0;       // preparation/open parameter
  std::uint64_t repeat_reconcile = 1; // bench parameter

  // preparation timing (persisted in prep.meta)
  double prep_total_ms = 0;
  double prep_open_ms = 0;
  double prep_populate_ms = 0;
  double prep_commit_ms = 0;
  double prep_seal_ms = 0;
  double prep_expected_ms = 0;
  double prep_serialize_ms = 0;

  // benchmark timing
  double open_ms = 0;
  double build_ms = 0;
  double reconcile_ms = 0;      // includes decode_sort_ms
  double decode_sort_ms = 0;
  double total_bench_ms = 0;

  // protocol stats (from the most recent reconcile; deterministic for a fixed dataset)
  std::size_t msg_count = 0;
  std::size_t bytes_a_to_b = 0;
  std::size_t bytes_b_to_a = 0;

  // results
  std::size_t have_count = 0;
  std::size_t need_count = 0;

  // on-disk sizes (bytes)
  std::uint64_t A_apparent_bytes = 0;
  std::uint64_t A_alloc_bytes = 0;
  std::uint64_t A_used_bytes_est = 0;
  std::uint64_t B_apparent_bytes = 0;
  std::uint64_t B_alloc_bytes = 0;
  std::uint64_t B_used_bytes_est = 0;

  // layout
  DbLayout layoutA{};
  DbLayout layoutB{};

  // memory
  std::size_t rss_kb_before = 0;
  std::size_t rss_kb_after = 0;
};

struct Expected {
  std::vector<std::uint64_t> have_u64;
  std::vector<std::uint64_t> need_u64;
};

static inline bool is_all_digits(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s) if (c < '0' || c > '9') return false;
  return true;
}

static inline std::pair<std::string, int> scenario_family_i(const std::string& name) {
  // Split "foo_bar_12" => ("foo_bar", 12). If no numeric suffix, i=0 and family=name.
  const auto pos = name.rfind('_');
  if (pos == std::string::npos) return {name, 0};
  const std::string suf = name.substr(pos + 1);
  if (!is_all_digits(suf)) return {name, 0};
  int i = 0;
  for (char c : suf) i = i * 10 + int(c - '0');
  return {name.substr(0, pos), i};
}

static inline void print_row_csv(const Scenario& sc,
                                int magnitude,
                                Backend backend,
                                const Expected& exp,
                                const Metrics& m) {
  const auto [family, si] = scenario_family_i(sc.name);
  const std::uint64_t expected_have = std::uint64_t(exp.have_u64.size());
  const std::uint64_t expected_need = std::uint64_t(exp.need_u64.size());

  std::cout
    << sc.name << ',' << family << ',' << si << ',' << magnitude << ','
    << backend_name(backend) << ','
    << obefore_order_name(m.obefore_order) << ',' << m.commit_every << ','
    << m.mapsize_mb << ',' << m.repeat_reconcile << ','

    // scenario parameters (shape)
    << sc.slice_begin_ts << ',' << sc.slice_end_ts << ',' << sc.step_in_slice << ','
    << sc.n_common_in_slice << ',' << sc.n_a_only_in_slice << ',' << sc.n_b_only_in_slice << ','
    << sc.n_common_outside_before << ',' << sc.n_common_outside_after << ','
    << sc.n_a_only_outside_before << ',' << sc.n_a_only_outside_after << ','
    << sc.n_b_only_outside_before << ',' << sc.n_b_only_outside_after << ','

    // observed sizes
    << m.fullA << ',' << m.fullB << ',' << m.sliceA << ',' << m.sliceB << ','

    // expected correctness denominators + observed results
    << expected_have << ',' << expected_need << ','
    << m.have_count << ',' << m.need_count << ','

    // timings
    << std::fixed << std::setprecision(3)
    << m.prep_total_ms << ',' << m.prep_open_ms << ',' << m.prep_populate_ms << ',' << m.prep_commit_ms << ','
    << m.prep_seal_ms << ',' << m.prep_expected_ms << ',' << m.prep_serialize_ms << ','
    << m.open_ms << ',' << m.build_ms << ',' << m.reconcile_ms << ',' << m.decode_sort_ms << ',' << m.total_bench_ms << ','

    // protocol
    << m.msg_count << ',' << m.bytes_a_to_b << ',' << m.bytes_b_to_a << ','

    // disk
    << m.A_apparent_bytes << ',' << m.A_alloc_bytes << ',' << m.A_used_bytes_est << ','
    << m.B_apparent_bytes << ',' << m.B_alloc_bytes << ',' << m.B_used_bytes_est << ','

    // layout
    << m.layoutA.depth << ',' << m.layoutA.branch_pages << ',' << m.layoutA.leaf_pages << ','
    << m.layoutA.overflow_pages << ',' << m.layoutA.entries << ',' << m.layoutA.last_pgno << ',' << m.layoutA.page_size << ','
    << m.layoutB.depth << ',' << m.layoutB.branch_pages << ',' << m.layoutB.leaf_pages << ','
    << m.layoutB.overflow_pages << ',' << m.layoutB.entries << ',' << m.layoutB.last_pgno << ',' << m.layoutB.page_size << ','

    // memory
    << m.rss_kb_before << ',' << m.rss_kb_after
    << '\n';
}

static inline void print_header_csv() {
  std::cout
    << "scenario,scenario_family,scenario_i,magnitude,backend,obefore_order,commit_every,mapsize_mb,repeat_reconcile,"
    << "slice_begin_ts,slice_end_ts,step_in_slice,"
    << "n_common_in_slice,n_a_only_in_slice,n_b_only_in_slice,"
    << "n_common_outside_before,n_common_outside_after,n_a_only_outside_before,n_a_only_outside_after,n_b_only_outside_before,n_b_only_outside_after,"
    << "fullA,fullB,sliceA,sliceB,expected_have,expected_need,have,need,"
    << "prep_total_ms,prep_open_ms,prep_populate_ms,prep_commit_ms,prep_seal_ms,prep_expected_ms,prep_serialize_ms,"
    << "open_ms,build_ms,reconcile_ms,decode_sort_ms,total_bench_ms,"
    << "msg_count,bytes_a_to_b,bytes_b_to_a,"
    << "A_apparent_bytes,A_alloc_bytes,A_used_bytes_est,B_apparent_bytes,B_alloc_bytes,B_used_bytes_est,"
    << "A_depth,A_branch_pages,A_leaf_pages,A_overflow_pages,A_entries,A_last_pgno,A_page_size,"
    << "B_depth,B_branch_pages,B_leaf_pages,B_overflow_pages,B_entries,B_last_pgno,B_page_size,"
    << "rss_kb_before,rss_kb_after\n" << std::flush;
}

// ---------- Dataset generation helpers (shared) ----------

static inline void gen_insert_slice_common(const Scenario& sc,
                                           const std::function<void(std::uint64_t, const IdBytes&)>& emit) {
  for (std::uint64_t i = 0; i < sc.n_common_in_slice; ++i) {
    const std::uint64_t ts = sc.slice_begin_ts + i * sc.step_in_slice;
    emit(ts, rbsr_test::pack_id_le(sc.base_common + i));
  }
}

static inline void gen_insert_slice_only(const Scenario& sc,
                                         std::uint64_t start_after_common,
                                         std::uint64_t n,
                                         std::uint64_t base,
                                         const std::function<void(std::uint64_t, const IdBytes&)>& emit) {
  for (std::uint64_t j = 0; j < n; ++j) {
    const std::uint64_t ts = sc.slice_begin_ts + (start_after_common + j) * sc.step_in_slice;
    emit(ts, rbsr_test::pack_id_le(base + j));
  }
}

static inline void gen_insert_outside_before(const Scenario& sc,
                                             bool is_peer_a,
                                             OutsideBeforeOrder order,
                                             const std::function<void(std::uint64_t, const IdBytes&)>& emit) {
  // Common outside-before
  const auto emit_common = [&](std::uint64_t idx) {
    const std::uint64_t ts = sc.slice_begin_ts - 1 - idx;
    emit(ts, rbsr_test::pack_id_le(sc.base_out_common + idx));
  };
  if (order == OutsideBeforeOrder::Desc) {
    for (std::uint64_t i = 0; i < sc.n_common_outside_before; ++i) emit_common(i);
  } else {
    for (std::uint64_t i = sc.n_common_outside_before; i-- > 0;) emit_common(i);
  }

  // A-only / B-only outside-before
  const std::uint64_t n_only = is_peer_a ? sc.n_a_only_outside_before : sc.n_b_only_outside_before;
  const std::uint64_t base   = is_peer_a ? sc.base_out_a : sc.base_out_b;
  const auto emit_only = [&](std::uint64_t idx) {
    const std::uint64_t ts = sc.slice_begin_ts - 1 - (sc.n_common_outside_before + idx);
    emit(ts, rbsr_test::pack_id_le(base + idx));
  };

  if (order == OutsideBeforeOrder::Desc) {
    for (std::uint64_t i = 0; i < n_only; ++i) emit_only(i);
  } else {
    for (std::uint64_t i = n_only; i-- > 0;) emit_only(i);
  }
}

static inline void gen_insert_outside_after(const Scenario& sc,
                                            bool is_peer_a,
                                            const std::function<void(std::uint64_t, const IdBytes&)>& emit) {
  // Common outside-after
  const std::uint64_t base_ts = sc.slice_end_ts;
  for (std::uint64_t i = 0; i < sc.n_common_outside_after; ++i) {
    const std::uint64_t ts = base_ts + i;
    emit(ts, rbsr_test::pack_id_le(sc.base_out_common + sc.n_common_outside_before + i));
  }

  // A-only / B-only outside-after
  const std::uint64_t n_only = is_peer_a ? sc.n_a_only_outside_after : sc.n_b_only_outside_after;
  const std::uint64_t base   = is_peer_a ? (sc.base_out_a + sc.n_a_only_outside_before)
                                         : (sc.base_out_b + sc.n_b_only_outside_before);
  for (std::uint64_t i = 0; i < n_only; ++i) {
    const std::uint64_t ts = base_ts + (sc.n_common_outside_after + i);
    emit(ts, rbsr_test::pack_id_le(base + i));
  }
}

static inline void populate_events_peer_vector(negentropy::storage::Vector& store,
                                               const Scenario& sc, bool is_peer_a,
                                               OutsideBeforeOrder ob_order) {
  auto insert_v = [&](std::uint64_t ts, const IdBytes& id){
    const std::string_view idsv(reinterpret_cast<const char*>(id.data()), id.size());
    store.insert(ts, idsv);
  };

  gen_insert_slice_common(sc, insert_v);

  const std::uint64_t start_after_common = sc.n_common_in_slice;
  if (is_peer_a) {
    gen_insert_slice_only(sc, start_after_common, sc.n_a_only_in_slice, sc.base_a_only, insert_v);
  } else {
    gen_insert_slice_only(sc, start_after_common, sc.n_b_only_in_slice, sc.base_b_only, insert_v);
  }

  gen_insert_outside_before(sc, is_peer_a, ob_order, insert_v);
  gen_insert_outside_after(sc, is_peer_a, insert_v);
}

static inline Expected expected_for_slice(const Scenario& sc) {
  Expected e;
  e.have_u64.reserve(sc.n_a_only_in_slice);
  e.need_u64.reserve(sc.n_b_only_in_slice);

  for (std::uint64_t j = 0; j < sc.n_a_only_in_slice; ++j) e.have_u64.push_back(sc.base_a_only + j);
  for (std::uint64_t j = 0; j < sc.n_b_only_in_slice; ++j) e.need_u64.push_back(sc.base_b_only + j);

  std::sort(e.have_u64.begin(), e.have_u64.end());
  std::sort(e.need_u64.begin(), e.need_u64.end());
  return e;
}

static inline void run_exchange_protocol(negentropy::StorageBase& storeA,
                                         negentropy::StorageBase& storeB,
                                         std::vector<std::string>& have,
                                         std::vector<std::string>& need,
                                         Metrics& m) {
  have.clear();
  need.clear();

  auto neA = negentropy::Negentropy(storeA);
  auto neB = negentropy::Negentropy(storeB);

  std::string msg = neA.initiate();
  m.msg_count = 0;
  m.bytes_a_to_b = 0;
  m.bytes_b_to_a = 0;

  while (true) {
    ++m.msg_count;
    m.bytes_a_to_b += msg.size();

    std::string response = neB.reconcile(msg);
    m.bytes_b_to_a += response.size();

    auto next = neA.reconcile(response, have, need);
    if (!next) break;
    msg = *next;
  }
}

static inline void decode_sort_ids(const std::vector<std::string>& have,
                                   const std::vector<std::string>& need,
                                   std::vector<std::uint64_t>& have_u,
                                   std::vector<std::uint64_t>& need_u,
                                   Metrics& m) {
  have_u.clear();
  need_u.clear();
  have_u.reserve(have.size());
  need_u.reserve(need.size());

  for (const auto& id : have) have_u.push_back(rbsr_test::unpack_id_le(id));
  for (const auto& id : need) need_u.push_back(rbsr_test::unpack_id_le(id));

  std::sort(have_u.begin(), have_u.end());
  std::sort(need_u.begin(), need_u.end());

  m.have_count = have_u.size();
  m.need_count = need_u.size();
}

static inline void reconcile_collect_ids(negentropy::StorageBase& storeA,
                                        negentropy::StorageBase& storeB,
                                        std::vector<std::uint64_t>& have_u,
                                        std::vector<std::uint64_t>& need_u,
                                        Metrics& m) {
  // Convenience wrapper used in init mode when computing persisted expected results.
  std::vector<std::string> have_s, need_s;
  run_exchange_protocol(storeA, storeB, have_s, need_s, m);
  decode_sort_ids(have_s, need_s, have_u, need_u, m);
}

static inline void assert_expected(const std::vector<std::uint64_t>& have_u,
                                   const std::vector<std::uint64_t>& need_u,
                                   const Expected& e,
                                   const Scenario& sc,
                                   Backend b) {
  if (have_u != e.have_u64 || need_u != e.need_u64) {
    auto dump = [&](const char* which,
                    const std::vector<std::uint64_t>& got,
                    const std::vector<std::uint64_t>& expv) {
      std::cerr << "Mismatch " << which << " for backend=" << backend_name(b)
                << " scenario=" << sc.name
                << " got=" << got.size() << " expected=" << expv.size() << "\n";
      const std::size_t n = std::min(got.size(), expv.size());
      for (std::size_t i = 0; i < n; ++i) {
        if (got[i] != expv[i]) {
          std::cerr << "  first diff at i=" << i << " got=" << got[i] << " expected=" << expv[i] << "\n";
          break;
        }
      }
      auto tail = [&](const char* label, const std::vector<std::uint64_t>& v) {
        std::cerr << "  " << label << " [";
        const std::size_t k = std::min<std::size_t>(8, v.size());
        for (std::size_t i = 0; i < k; ++i) {
          if (i) std::cerr << ",";
          std::cerr << v[i];
        }
        if (v.size() > k) std::cerr << ",...";
        std::cerr << "]\n";
      };
      tail("got", got);
      tail("expected", expv);
    };

    if (have_u != e.have_u64) dump("HAVE", have_u, e.have_u64);
    if (need_u != e.need_u64) dump("NEED", need_u, e.need_u64);
    throw std::runtime_error("have/need mismatch");
  }
}

// ---------- LMDB stat helpers ----------

static inline DbLayout read_layout_from_lmdb(lmdb::env& env, lmdb::txn& txn, const lmdb::dbi& dbi) {
  DbLayout out;
  MDB_stat dbst{};
  mdb_stat(txn.handle(), dbi.handle(), &dbst);
  out.depth = dbst.ms_depth;
  out.branch_pages = dbst.ms_branch_pages;
  out.leaf_pages = dbst.ms_leaf_pages;
  out.overflow_pages = dbst.ms_overflow_pages;
  out.entries = dbst.ms_entries;

  MDB_stat envst{};
  mdb_env_stat(env.handle(), &envst);
  out.page_size = envst.ms_psize;

  MDB_envinfo info{};
  mdb_env_info(env.handle(), &info);
  out.last_pgno = info.me_last_pgno;
  return out;
}

// Boundaries used for all backends.
static inline std::pair<rbsr_test::KeyBytes, rbsr_test::KeyBytes>
make_slice_boundaries(const Scenario& sc) {
  return {rbsr_test::make_boundary_key(sc.slice_begin_ts),
          rbsr_test::make_boundary_key(sc.slice_end_ts)};
}

static inline std::pair<std::uint64_t, std::uint64_t>
compute_slice_indices(negentropy::StorageBase& store,
                      std::uint64_t slice_begin_ts,
                      std::uint64_t slice_end_ts) {
  const negentropy::Bound lower(slice_begin_ts);
  const negentropy::Bound upper(slice_end_ts);
  const auto n = std::uint64_t(store.size());
  const auto lo = std::uint64_t(store.findLowerBound(0, size_t(n), lower));
  const auto hi = std::uint64_t(store.findLowerBound(0, size_t(n), upper));
  return {lo, hi};
}

// ---------- Serialization for Vector prepared state ----------

static inline void write_vector_items(const std::string& path, negentropy::storage::Vector& store) {
  // negentropy::storage::Vector requires sealing before any query APIs (size/getItem).
  // Callers may have sealed already (Vector::seal() throws if called twice), so only seal when needed.
  try {
    (void)store.size();
  } catch (const std::runtime_error& e) {
    const std::string_view msg(e.what());
    if (msg.find("not sealed") != std::string_view::npos) {
      store.seal();
    } else {
      throw;
    }
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("failed to open for write: " + path);

  const std::uint64_t n = store.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  for (std::uint64_t i = 0; i < n; ++i) {
    const negentropy::Item& it = store.getItem(size_t(i));
    rbsr_test::KeyBytes kb{};
    rbsr_test::store_be64(kb.data(), it.timestamp);
    std::memcpy(kb.data() + rbsr_test::TS_SIZE, it.id, rbsr_test::ID_SIZE);
    out.write(reinterpret_cast<const char*>(kb.data()), kb.size());
  }
}

static inline void read_vector_items(const std::string& path, negentropy::storage::Vector& store) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open for read: " + path);

  std::uint64_t n = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (!in) throw std::runtime_error("bad vector file header: " + path);

  // Avoid depending on a Vector::clear() API; reassign to an empty store.
  store = negentropy::storage::Vector{};
  for (std::uint64_t i = 0; i < n; ++i) {
    rbsr_test::KeyBytes kb{};
    in.read(reinterpret_cast<char*>(kb.data()), kb.size());
    if (!in) throw std::runtime_error("bad vector file body: " + path);
    const std::uint64_t ts = rbsr_test::load_be64(kb.data());
    const std::string_view idsv(reinterpret_cast<const char*>(kb.data() + rbsr_test::TS_SIZE), rbsr_test::ID_SIZE);
    store.insert(ts, idsv);
  }

  // Seal after loading so that size()/getItem()/findLowerBound() are valid.
  store.seal();
}


// ---------- Expected HAVE/NEED persistence (to keep bench mode independent of generator changes) ----------

static inline std::string join_path(const std::string& a, const std::string& b) {
  return (std::filesystem::path(a) / b).string();
}

static inline void ensure_clean_dir(const std::string& dir) {
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir);
}

static inline std::string expected_dir(const std::string& root, const Scenario& sc) {
  return join_path(join_path(root, "_expected"), sc.name);
}

static inline std::uint64_t scenario_sig64(const Scenario& sc) {
  // FNV-1a 64-bit over all scenario-defining numeric fields.
  std::uint64_t h = 1469598103934665603ull;
  auto mix_u64 = [&](std::uint64_t x) {
    for (int i = 0; i < 8; ++i) {
      h ^= (x >> (8 * i)) & 0xffull;
      h *= 1099511628211ull;
    }
  };
  mix_u64(sc.slice_begin_ts);
  mix_u64(sc.slice_end_ts);
  mix_u64(sc.step_in_slice);

  mix_u64(sc.n_common_in_slice);
  mix_u64(sc.n_a_only_in_slice);
  mix_u64(sc.n_b_only_in_slice);

  mix_u64(sc.n_common_outside_before);
  mix_u64(sc.n_common_outside_after);
  mix_u64(sc.n_a_only_outside_before);
  mix_u64(sc.n_a_only_outside_after);
  mix_u64(sc.n_b_only_outside_before);
  mix_u64(sc.n_b_only_outside_after);

  mix_u64(sc.base_common);
  mix_u64(sc.base_a_only);
  mix_u64(sc.base_b_only);
  mix_u64(sc.base_out_common);
  mix_u64(sc.base_out_a);
  mix_u64(sc.base_out_b);
  return h;
}

static inline void write_u64_vec(const std::string& path, const std::vector<std::uint64_t>& v) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("failed to open for write: " + path);
  const std::uint64_t n = std::uint64_t(v.size());
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  if (n) out.write(reinterpret_cast<const char*>(v.data()), std::streamsize(n * sizeof(std::uint64_t)));
  if (!out) throw std::runtime_error("failed writing: " + path);
}

static inline std::vector<std::uint64_t> read_u64_vec(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open for read: " + path);
  std::uint64_t n = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (!in) throw std::runtime_error("bad header: " + path);
  std::vector<std::uint64_t> v;
  v.resize(std::size_t(n));
  if (n) {
    in.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(std::uint64_t)));
    if (!in) throw std::runtime_error("bad body: " + path);
  }
  return v;
}

static inline void write_expected(const std::string& root, const Scenario& sc, const Expected& e) {
  const std::string dir = expected_dir(root, sc);
  ensure_clean_dir(dir);
  // meta (scenario signature)
  {
    std::ofstream meta(join_path(dir, "meta.txt"), std::ios::trunc);
    if (!meta) throw std::runtime_error("failed to open meta for write");
    meta << std::hex << scenario_sig64(sc) << "\n";
  }
  write_u64_vec(join_path(dir, "have_u64.bin"), e.have_u64);
  write_u64_vec(join_path(dir, "need_u64.bin"), e.need_u64);
}

static inline std::optional<Expected> read_expected(const std::string& root, const Scenario& sc) {
  const std::string dir = expected_dir(root, sc);
  if (!std::filesystem::exists(dir)) return std::nullopt;
  // meta check (helps catch stale prepared roots)
  {
    std::ifstream meta(join_path(dir, "meta.txt"));
    if (!meta) return std::nullopt;
    std::uint64_t on_disk = 0;
    meta >> std::hex >> on_disk;
    if (!meta) return std::nullopt;
    const std::uint64_t cur = scenario_sig64(sc);
    if (on_disk != cur) {
      std::ostringstream oss;
      oss << "prepared expected mismatch for scenario '" << sc.name
          << "': signature differs (on-disk=" << std::hex << on_disk
          << " current=" << cur << "). Delete the prepared root and rerun --mode init.";
      throw std::runtime_error(oss.str());
    }
  }
  Expected e;
  e.have_u64 = read_u64_vec(join_path(dir, "have_u64.bin"));
  e.need_u64 = read_u64_vec(join_path(dir, "need_u64.bin"));
  std::sort(e.have_u64.begin(), e.have_u64.end());
  std::sort(e.need_u64.begin(), e.need_u64.end());
  return e;
}

// ---------- Prepared directory layout ----------

static inline std::string scenario_root(const std::string& root, const Scenario& sc, Backend b) {
  return (std::filesystem::path(root) / backend_name(b) / sc.name).string();
}

static inline std::string peer_path(const std::string& root, const Scenario& sc, Backend b, const char* peer) {
  return (std::filesystem::path(scenario_root(root, sc, b)) / peer).string();
}

struct PrepMeta {
  OutsideBeforeOrder obefore_order = OutsideBeforeOrder::Desc;
  std::uint64_t commit_every = 0;
  std::uint64_t mapsize_mb = 0;
  double prep_total_ms = 0;
  double prep_open_ms = 0;
  double prep_populate_ms = 0;
  double prep_commit_ms = 0;
  double prep_seal_ms = 0;
  double prep_expected_ms = 0;
  double prep_serialize_ms = 0;
};

static inline std::uint64_t parse_u64_strict(const std::string& s) {
  if (s.empty()) throw std::runtime_error("bad integer (empty)");
  std::uint64_t v = 0;
  for (char c : s) {
    if (c < '0' || c > '9') throw std::runtime_error("bad integer: " + s);
    v = v * 10 + std::uint64_t(c - '0');
  }
  return v;
}

static inline double parse_double_strict(const std::string& s) {
  std::size_t idx = 0;
  double v = 0;
  try {
    v = std::stod(s, &idx);
  } catch (...) {
    throw std::runtime_error("bad double: " + s);
  }
  if (idx != s.size()) throw std::runtime_error("bad double: " + s);
  return v;
}

static inline OutsideBeforeOrder parse_obefore_strict(const std::string& s) {
  if (s == "desc") return OutsideBeforeOrder::Desc;
  if (s == "asc") return OutsideBeforeOrder::Asc;
  throw std::runtime_error("bad obefore_order: " + s);
}

static inline std::string prep_meta_path(const std::string& root, const Scenario& sc, Backend b) {
  return (std::filesystem::path(scenario_root(root, sc, b)) / "prep.meta").string();
}

static inline void write_prep_meta(const std::string& root, const Scenario& sc, Backend b,
                                   const PrepMeta& pm) {
  const std::string dir = scenario_root(root, sc, b);
  std::filesystem::create_directories(dir);
  const std::string meta = prep_meta_path(root, sc, b);
  std::ofstream out(meta, std::ios::trunc);
  if (!out) throw std::runtime_error("failed to write prep.meta");

  out << "obefore_order=" << obefore_order_name(pm.obefore_order) << "\n";
  out << "commit_every=" << pm.commit_every << "\n";
  out << "mapsize_mb=" << pm.mapsize_mb << "\n";
  out << std::fixed << std::setprecision(6);
  out << "prep_total_ms=" << pm.prep_total_ms << "\n";
  out << "prep_open_ms=" << pm.prep_open_ms << "\n";
  out << "prep_populate_ms=" << pm.prep_populate_ms << "\n";
  out << "prep_commit_ms=" << pm.prep_commit_ms << "\n";
  out << "prep_seal_ms=" << pm.prep_seal_ms << "\n";
  out << "prep_expected_ms=" << pm.prep_expected_ms << "\n";
  out << "prep_serialize_ms=" << pm.prep_serialize_ms << "\n";
}

static inline PrepMeta read_prep_meta(const std::string& root, const Scenario& sc, Backend b) {
  const std::string meta = prep_meta_path(root, sc, b);
  std::ifstream in(meta);
  if (!in) throw std::runtime_error("missing prep.meta: " + meta + " (run --mode init)");

  std::unordered_map<std::string, std::string> kv;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto pos = line.find('=');
    if (pos == std::string::npos) throw std::runtime_error("bad prep.meta line: " + line);
    kv[line.substr(0, pos)] = line.substr(pos + 1);
  }

  auto get = [&](const char* k) -> const std::string& {
    auto it = kv.find(k);
    if (it == kv.end()) throw std::runtime_error(std::string("prep.meta missing key: ") + k);
    return it->second;
  };

  PrepMeta pm;
  pm.obefore_order = parse_obefore_strict(get("obefore_order"));
  pm.commit_every = parse_u64_strict(get("commit_every"));
  pm.mapsize_mb = parse_u64_strict(get("mapsize_mb"));
  pm.prep_total_ms = parse_double_strict(get("prep_total_ms"));
  pm.prep_open_ms = parse_double_strict(get("prep_open_ms"));
  pm.prep_populate_ms = parse_double_strict(get("prep_populate_ms"));
  pm.prep_commit_ms = parse_double_strict(get("prep_commit_ms"));
  pm.prep_seal_ms = parse_double_strict(get("prep_seal_ms"));
  pm.prep_expected_ms = parse_double_strict(get("prep_expected_ms"));
  pm.prep_serialize_ms = parse_double_strict(get("prep_serialize_ms"));
  return pm;
}

// ---------- Preparation (outside benchmark timing) ----------

struct LMDBBatchWriter {
  rbsr_test::PeerDB& P;
  lmdb::txn txn;
  std::uint64_t pending = 0;
  std::uint64_t commit_every = 0;
  double commit_ms = 0;

  LMDBBatchWriter(rbsr_test::PeerDB& peer, std::uint64_t commit_every_)
      : P(peer), txn(lmdb::txn::begin(P.env)), pending(0), commit_every(commit_every_) {}

  void maybe_commit() {
    if (commit_every == 0) return;
    if (pending < commit_every) return;
    Timer t;
    txn.commit();
    commit_ms += t.ms();
    txn = lmdb::txn::begin(P.env);
    pending = 0;
  }

  void put(std::uint64_t ts, const IdBytes& id) {
    rbsr_test::insert_item(txn, P.dbi, ts, id, MDB_NOOVERWRITE);
    ++pending;
    maybe_commit();
  }

  void finish() {
    Timer t;
    txn.commit();
    commit_ms += t.ms();
  }
};

struct BTreeBatchWriter {
  lmdb::env& env;
  lmdb::dbi dbi;
  std::uint64_t tree_id;
  std::uint64_t commit_every;
  lmdb::txn txn;
  // BTreeLMDB is not assignable (and may not be movable), so keep it behind a pointer.
  std::unique_ptr<negentropy::storage::BTreeLMDB> store;
  std::uint64_t pending = 0;
  double commit_ms = 0;

  BTreeBatchWriter(lmdb::env& e, lmdb::dbi d, std::uint64_t tid, std::uint64_t ce)
      : env(e), dbi(d), tree_id(tid), commit_every(ce),
        txn(lmdb::txn::begin(env)), store(std::make_unique<negentropy::storage::BTreeLMDB>(txn, dbi, tree_id)) {}

  void maybe_commit() {
    if (commit_every == 0) return;
    if (pending < commit_every) return;
    Timer t;
    store->flush();
    txn.commit();
    commit_ms += t.ms();
    txn = lmdb::txn::begin(env);
    store.reset(new negentropy::storage::BTreeLMDB(txn, dbi, tree_id));
    pending = 0;
  }

  void put(std::uint64_t ts, const IdBytes& id) {
    const std::string_view idsv(reinterpret_cast<const char*>(id.data()), id.size());
    store->insert(ts, idsv);
    ++pending;
    maybe_commit();
  }

  void finish() {
    Timer t;
    store->flush();
    txn.commit();
    commit_ms += t.ms();
  }
};

static inline void prepare_case(const Scenario& sc,
                                Backend backend,
                                OutsideBeforeOrder ob_order,
                                std::uint64_t commit_every,
                                std::uint64_t mapsize_mb,
                                const std::string& root) {
  Timer t_total_prep;

  PrepMeta pm;
  pm.obefore_order = ob_order;
  pm.commit_every = commit_every;
  pm.mapsize_mb = mapsize_mb;

  if (backend == Backend::Vector) {
    const std::string dir = scenario_root(root, sc, backend);

    {
      Timer t;
      ensure_clean_dir(dir);
      pm.prep_open_ms = t.ms();
    }

    negentropy::storage::Vector A;
    negentropy::storage::Vector B;

    {
      Timer t;
      populate_events_peer_vector(A, sc, /*is_peer_a=*/true, ob_order);
      populate_events_peer_vector(B, sc, /*is_peer_a=*/false, ob_order);
      pm.prep_populate_ms = t.ms();
    }

    // Seal is required before any query APIs (size/getItem/findLowerBound).
    {
      Timer t;
      A.seal();
      B.seal();
      pm.prep_seal_ms = t.ms();
    }

    // Compute and persist expected HAVE/NEED for this scenario based on the prepared data.
    // This avoids relying on analytic expectations that can drift if scenario generation changes.
    {
      Timer t;
      const auto [loA, hiA] = compute_slice_indices(A, sc.slice_begin_ts, sc.slice_end_ts);
      const auto [loB, hiB] = compute_slice_indices(B, sc.slice_begin_ts, sc.slice_end_ts);
      IndexSlice sliceA(A, loA, hiA);
      IndexSlice sliceB(B, loB, hiB);

      Metrics dummy;
      std::vector<std::uint64_t> have_u;
      std::vector<std::uint64_t> need_u;
      reconcile_collect_ids(sliceA, sliceB, have_u, need_u, dummy);

      Expected e;
      e.have_u64 = std::move(have_u);
      e.need_u64 = std::move(need_u);
      std::sort(e.have_u64.begin(), e.have_u64.end());
      std::sort(e.need_u64.begin(), e.need_u64.end());
      write_expected(root, sc, e);

      pm.prep_expected_ms = t.ms();
    }

    {
      Timer t;
      write_vector_items(join_path(dir, "A.bin"), A);
      write_vector_items(join_path(dir, "B.bin"), B);
      pm.prep_serialize_ms = t.ms();
    }

    pm.prep_total_ms = t_total_prep.ms();
    write_prep_meta(root, sc, backend, pm);
    return;
  }

  if (backend == Backend::AELMDBSlice) {
    const std::string pathA = peer_path(root, sc, backend, "A");
    const std::string pathB = peer_path(root, sc, backend, "B");

    Timer t_open;
    auto A = rbsr_test::PeerDB::create_fresh(pathA, "events", mapsize_mb);
    auto B = rbsr_test::PeerDB::create_fresh(pathB, "events", mapsize_mb);
    pm.prep_open_ms = t_open.ms();

    LMDBBatchWriter wA(A, commit_every);
    LMDBBatchWriter wB(B, commit_every);

    Timer t_pop;

    gen_insert_slice_common(sc, [&](std::uint64_t ts, const IdBytes& id){ wA.put(ts, id); wB.put(ts, id); });

    const std::uint64_t start_after_common = sc.n_common_in_slice;
    gen_insert_slice_only(sc, start_after_common, sc.n_a_only_in_slice, sc.base_a_only,
      [&](std::uint64_t ts, const IdBytes& id){ wA.put(ts, id); });
    gen_insert_slice_only(sc, start_after_common, sc.n_b_only_in_slice, sc.base_b_only,
      [&](std::uint64_t ts, const IdBytes& id){ wB.put(ts, id); });

    gen_insert_outside_before(sc, /*is_peer_a=*/true, ob_order, [&](std::uint64_t ts, const IdBytes& id){ wA.put(ts, id); });
    gen_insert_outside_before(sc, /*is_peer_a=*/false, ob_order, [&](std::uint64_t ts, const IdBytes& id){ wB.put(ts, id); });

    gen_insert_outside_after(sc, /*is_peer_a=*/true, [&](std::uint64_t ts, const IdBytes& id){ wA.put(ts, id); });
    gen_insert_outside_after(sc, /*is_peer_a=*/false, [&](std::uint64_t ts, const IdBytes& id){ wB.put(ts, id); });

    wA.finish();
    wB.finish();

    pm.prep_populate_ms = t_pop.ms();
    pm.prep_commit_ms = wA.commit_ms + wB.commit_ms;

    pm.prep_total_ms = t_total_prep.ms();
    write_prep_meta(root, sc, backend, pm);
    return;
  }

  if (backend == Backend::BTreeLMDB) {
    const std::string pathA = peer_path(root, sc, backend, "A");
    const std::string pathB = peer_path(root, sc, backend, "B");

    Timer t_open;
    ensure_clean_dir(pathA);
    ensure_clean_dir(pathB);

    auto envA = lmdb::env::create();
    auto envB = lmdb::env::create();
    envA.set_max_dbs(16);
    envB.set_max_dbs(16);
    envA.set_mapsize(std::uint64_t(mapsize_mb) * 1024ull * 1024ull);
    envB.set_mapsize(std::uint64_t(mapsize_mb) * 1024ull * 1024ull);
    envA.open(pathA.c_str(), 0);
    envB.open(pathB.c_str(), 0);

    lmdb::dbi dbiA{0}, dbiB{0};
    {
      auto txnA = lmdb::txn::begin(envA);
      auto txnB = lmdb::txn::begin(envB);
      dbiA = negentropy::storage::BTreeLMDB::setupDB(txnA, "btree");
      dbiB = negentropy::storage::BTreeLMDB::setupDB(txnB, "btree");
      txnA.commit();
      txnB.commit();
    }

    pm.prep_open_ms = t_open.ms();

    BTreeBatchWriter wA(envA, dbiA, 1, commit_every);
    BTreeBatchWriter wB(envB, dbiB, 1, commit_every);

    Timer t_pop;

    gen_insert_slice_common(sc, [&](std::uint64_t ts, const IdBytes& id){ wA.put(ts, id); wB.put(ts, id); });

    const std::uint64_t start_after_common = sc.n_common_in_slice;
    gen_insert_slice_only(sc, start_after_common, sc.n_a_only_in_slice, sc.base_a_only,
      [&](std::uint64_t ts, const IdBytes& id){ wA.put(ts, id); });
    gen_insert_slice_only(sc, start_after_common, sc.n_b_only_in_slice, sc.base_b_only,
      [&](std::uint64_t ts, const IdBytes& id){ wB.put(ts, id); });

    gen_insert_outside_before(sc, /*is_peer_a=*/true, ob_order, [&](std::uint64_t ts, const IdBytes& id){ wA.put(ts, id); });
    gen_insert_outside_before(sc, /*is_peer_a=*/false, ob_order, [&](std::uint64_t ts, const IdBytes& id){ wB.put(ts, id); });

    gen_insert_outside_after(sc, /*is_peer_a=*/true, [&](std::uint64_t ts, const IdBytes& id){ wA.put(ts, id); });
    gen_insert_outside_after(sc, /*is_peer_a=*/false, [&](std::uint64_t ts, const IdBytes& id){ wB.put(ts, id); });

    wA.finish();
    wB.finish();

    pm.prep_populate_ms = t_pop.ms();
    pm.prep_commit_ms = wA.commit_ms + wB.commit_ms;

    pm.prep_total_ms = t_total_prep.ms();
    write_prep_meta(root, sc, backend, pm);
    return;
  }

  throw std::runtime_error("prepare_case: backend not supported");
}

// ---------- Benchmark (negentropy only; assumes prepared state exists) ----------

static inline void bench_case(const Scenario& sc,
                              const Expected& exp,
                              int magnitude,
                              Backend backend,
                              OutsideBeforeOrder ob_order,
                              std::uint64_t commit_every,
                              std::uint64_t mapsize_mb,
                              std::uint64_t repeat_reconcile,
                              const std::string& root) {
  Metrics m;
  m.rss_kb_before = read_rss_kb_linux();

  // Load preparation metadata (also captures the truth for obefore/commit_every used during init).
  const PrepMeta pm = read_prep_meta(root, sc, backend);
  if (pm.obefore_order != ob_order) {
    throw std::runtime_error("bench obefore_order differs from prepared prep.meta for scenario '" + sc.name + "'");
  }
  if (pm.commit_every != commit_every) {
    throw std::runtime_error("bench commit_every differs from prepared prep.meta for scenario '" + sc.name + "'");
  }
  if (pm.mapsize_mb != mapsize_mb) {
    throw std::runtime_error("bench mapsize_mb differs from prepared prep.meta for scenario '" + sc.name + "'");
  }

  m.obefore_order = pm.obefore_order;
  m.commit_every = pm.commit_every;
  m.mapsize_mb = pm.mapsize_mb;
  m.repeat_reconcile = (repeat_reconcile == 0 ? 1 : repeat_reconcile);
  m.prep_total_ms = pm.prep_total_ms;
  m.prep_open_ms = pm.prep_open_ms;
  m.prep_populate_ms = pm.prep_populate_ms;
  m.prep_commit_ms = pm.prep_commit_ms;
  m.prep_seal_ms = pm.prep_seal_ms;
  m.prep_expected_ms = pm.prep_expected_ms;
  m.prep_serialize_ms = pm.prep_serialize_ms;

  std::vector<std::uint64_t> have_u, need_u;
  std::vector<std::string> have_s, need_s;

  const auto [begin_kb, end_kb] = make_slice_boundaries(sc);
  const std::string_view begin_sv(reinterpret_cast<const char*>(begin_kb.data()), begin_kb.size());
  const std::string_view end_sv(reinterpret_cast<const char*>(end_kb.data()), end_kb.size());

  if (backend == Backend::Vector) {
    const std::string dir = scenario_root(root, sc, backend);
    const auto Afile = std::filesystem::path(dir) / "A.bin";
    const auto Bfile = std::filesystem::path(dir) / "B.bin";

    // Timed open/load.
    negentropy::storage::Vector A;
    negentropy::storage::Vector B;
    {
      Timer t;
      read_vector_items(Afile.string(), A);
      read_vector_items(Bfile.string(), B);
      m.open_ms = t.ms();
    }

    m.fullA = A.size();
    m.fullB = B.size();

    {
      const DiskBytes da = disk_bytes_of(Afile);
      const DiskBytes db = disk_bytes_of(Bfile);
      m.A_apparent_bytes = da.apparent;
      m.A_alloc_bytes = da.allocated;
      m.A_used_bytes_est = da.apparent;
      m.B_apparent_bytes = db.apparent;
      m.B_alloc_bytes = db.allocated;
      m.B_used_bytes_est = db.apparent;
    }

    // Timed slice build.
    {
      Timer t;
      const auto [loA, hiA] = compute_slice_indices(A, sc.slice_begin_ts, sc.slice_end_ts);
      const auto [loB, hiB] = compute_slice_indices(B, sc.slice_begin_ts, sc.slice_end_ts);
      IndexSlice sliceA(A, loA, hiA);
      IndexSlice sliceB(B, loB, hiB);
      m.sliceA = sliceA.size();
      m.sliceB = sliceB.size();
      m.build_ms = t.ms();

      // Timed reconciliation (possibly repeated for averaging).
      const std::uint64_t reps = m.repeat_reconcile;
      double sum_total = 0.0;
      double sum_decode = 0.0;

      for (std::uint64_t rep = 0; rep < reps; ++rep) {
        Timer tr;
        run_exchange_protocol(sliceA, sliceB, have_s, need_s, m);
        Timer td;
        decode_sort_ids(have_s, need_s, have_u, need_u, m);
        sum_decode += td.ms();
        sum_total += tr.ms();
        if (rep == 0) {
          assert_expected(have_u, need_u, exp, sc, backend);
        }
      }

      m.decode_sort_ms = sum_decode / double(reps);
      m.reconcile_ms = sum_total / double(reps);
    }

    m.total_bench_ms = m.open_ms + m.build_ms + m.reconcile_ms;

    m.rss_kb_after = read_rss_kb_linux();
    print_row_csv(sc, magnitude, backend, exp, m);
    return;
  }

  if (backend == Backend::AELMDBSlice) {
    const std::string pathA = peer_path(root, sc, backend, "A");
    const std::string pathB = peer_path(root, sc, backend, "B");

    // Open read-only (timed).
    Timer t_open;
    auto envA = lmdb::env::create();
    auto envB = lmdb::env::create();
    envA.set_max_dbs(16);
    envB.set_max_dbs(16);
    envA.set_mapsize(std::uint64_t(mapsize_mb) * 1024ull * 1024ull);
    envB.set_mapsize(std::uint64_t(mapsize_mb) * 1024ull * 1024ull);
    envA.open(pathA.c_str(), MDB_RDONLY);
    envB.open(pathB.c_str(), MDB_RDONLY);

    auto txnA = lmdb::txn::begin(envA, nullptr, MDB_RDONLY);
    auto txnB = lmdb::txn::begin(envB, nullptr, MDB_RDONLY);

    const unsigned int dbi_flags = MDB_AGG_ENTRIES | MDB_AGG_HASHSUM | MDB_AGG_HASHSOURCE_FROM_KEY;
    auto dbiA = lmdb::dbi::open(txnA, "events", dbi_flags);
    auto dbiB = lmdb::dbi::open(txnB, "events", dbi_flags);

    // Cheap metadata reads.
    m.fullA = dbiA.totals(txnA).mv_agg_entries;
    m.fullB = dbiB.totals(txnB).mv_agg_entries;
    m.layoutA = read_layout_from_lmdb(envA, txnA, dbiA);
    m.layoutB = read_layout_from_lmdb(envB, txnB, dbiB);

    m.open_ms = t_open.ms();

    {
      const DiskBytes da = disk_bytes_of(pathA);
      const DiskBytes db = disk_bytes_of(pathB);
      m.A_apparent_bytes = da.apparent;
      m.A_alloc_bytes = da.allocated;
      m.B_apparent_bytes = db.apparent;
      m.B_alloc_bytes = db.allocated;

      if (m.layoutA.page_size) m.A_used_bytes_est = (m.layoutA.last_pgno + 1) * m.layoutA.page_size;
      if (m.layoutB.page_size) m.B_used_bytes_est = (m.layoutB.last_pgno + 1) * m.layoutB.page_size;
    }

    // Timed: slice wrapper creation (constructor + first size()).
    std::unique_ptr<negentropy::storage::SliceAELMDB> storeA;
    std::unique_ptr<negentropy::storage::SliceAELMDB> storeB;
    {
      Timer t;
      storeA = std::make_unique<negentropy::storage::SliceAELMDB>(txnA, dbiA,
        std::optional<std::string_view>{begin_sv},
        std::optional<std::string_view>{end_sv});
      storeB = std::make_unique<negentropy::storage::SliceAELMDB>(txnB, dbiB,
        std::optional<std::string_view>{begin_sv},
        std::optional<std::string_view>{end_sv});

      m.sliceA = storeA->size();
      m.sliceB = storeB->size();
      m.build_ms = t.ms();
    }

    // Timed reconciliation (possibly repeated for averaging).
    {
      const std::uint64_t reps = m.repeat_reconcile;
      double sum_total = 0.0;
      double sum_decode = 0.0;

      for (std::uint64_t rep = 0; rep < reps; ++rep) {
        Timer tr;
        run_exchange_protocol(*storeA, *storeB, have_s, need_s, m);
        Timer td;
        decode_sort_ids(have_s, need_s, have_u, need_u, m);
        sum_decode += td.ms();
        sum_total += tr.ms();
        if (rep == 0) {
          assert_expected(have_u, need_u, exp, sc, backend);
        }
      }

      m.decode_sort_ms = sum_decode / double(reps);
      m.reconcile_ms = sum_total / double(reps);
    }

    m.total_bench_ms = m.open_ms + m.build_ms + m.reconcile_ms;

    m.rss_kb_after = read_rss_kb_linux();
    print_row_csv(sc, magnitude, backend, exp, m);
    return;
  }

  if (backend == Backend::BTreeLMDB) {
    const std::string pathA = peer_path(root, sc, backend, "A");
    const std::string pathB = peer_path(root, sc, backend, "B");

    Timer t_open;
    auto envA = lmdb::env::create();
    auto envB = lmdb::env::create();
    envA.set_max_dbs(16);
    envB.set_max_dbs(16);
    envA.set_mapsize(std::uint64_t(mapsize_mb) * 1024ull * 1024ull);
    envB.set_mapsize(std::uint64_t(mapsize_mb) * 1024ull * 1024ull);
    envA.open(pathA.c_str(), MDB_RDONLY);
    envB.open(pathB.c_str(), MDB_RDONLY);

    auto txnA = lmdb::txn::begin(envA, nullptr, MDB_RDONLY);
    auto txnB = lmdb::txn::begin(envB, nullptr, MDB_RDONLY);

    auto dbiA = lmdb::dbi::open(txnA, "btree", 0);
    auto dbiB = lmdb::dbi::open(txnB, "btree", 0);

    negentropy::storage::BTreeLMDB fullStoreA(txnA, dbiA, 1);
    negentropy::storage::BTreeLMDB fullStoreB(txnB, dbiB, 1);

    m.fullA = fullStoreA.size();
    m.fullB = fullStoreB.size();
    m.layoutA = read_layout_from_lmdb(envA, txnA, dbiA);
    m.layoutB = read_layout_from_lmdb(envB, txnB, dbiB);

    m.open_ms = t_open.ms();

    // Disk bytes
    {
      const DiskBytes da = disk_bytes_of(pathA);
      const DiskBytes db = disk_bytes_of(pathB);
      m.A_apparent_bytes = da.apparent;
      m.A_alloc_bytes = da.allocated;
      m.B_apparent_bytes = db.apparent;
      m.B_alloc_bytes = db.allocated;
      if (m.layoutA.page_size) m.A_used_bytes_est = (m.layoutA.last_pgno + 1) * m.layoutA.page_size;
      if (m.layoutB.page_size) m.B_used_bytes_est = (m.layoutB.last_pgno + 1) * m.layoutB.page_size;
    }

    // Slice build
    Timer tb;
    const auto [loA, hiA] = compute_slice_indices(fullStoreA, sc.slice_begin_ts, sc.slice_end_ts);
    const auto [loB, hiB] = compute_slice_indices(fullStoreB, sc.slice_begin_ts, sc.slice_end_ts);
    IndexSlice sliceA(fullStoreA, loA, hiA);
    IndexSlice sliceB(fullStoreB, loB, hiB);
    m.sliceA = sliceA.size();
    m.sliceB = sliceB.size();
    m.build_ms = tb.ms();

    // Reconcile (possibly repeated for averaging)
    {
      const std::uint64_t reps = m.repeat_reconcile;
      double sum_total = 0.0;
      double sum_decode = 0.0;

      for (std::uint64_t rep = 0; rep < reps; ++rep) {
        Timer tr;
        run_exchange_protocol(sliceA, sliceB, have_s, need_s, m);
        Timer td;
        decode_sort_ids(have_s, need_s, have_u, need_u, m);
        sum_decode += td.ms();
        sum_total += tr.ms();
        if (rep == 0) {
          assert_expected(have_u, need_u, exp, sc, backend);
        }
      }

      m.decode_sort_ms = sum_decode / double(reps);
      m.reconcile_ms = sum_total / double(reps);
    }

    m.total_bench_ms = m.open_ms + m.build_ms + m.reconcile_ms;

    m.rss_kb_after = read_rss_kb_linux();
    print_row_csv(sc, magnitude, backend, exp, m);
    return;
  }

  throw std::runtime_error("bench_case: backend not supported");
}

// ---------- Scenarios ----------

static inline void ensure_slice_end_covers(Scenario& sc) {
  const std::uint64_t in_count_max =
      sc.n_common_in_slice + std::max(sc.n_a_only_in_slice, sc.n_b_only_in_slice);
  const std::uint64_t needed = sc.slice_begin_ts + (in_count_max + 2) * sc.step_in_slice;
  if (sc.slice_end_ts < needed) sc.slice_end_ts = needed;
}

static inline std::vector<Scenario> make_scenarios(int magnitude) {
  const std::uint64_t M = std::uint64_t(magnitude);
  const std::uint64_t Q = M * M;

  std::vector<Scenario> out;
  auto add = [&](Scenario sc) {
    ensure_slice_end_covers(sc);
    out.push_back(std::move(sc));
  };

  add(Scenario{
    .name = "baseline_dense",
    .slice_begin_ts = 10'000,
    .slice_end_ts = 0,
    .step_in_slice = 1,
    .n_common_in_slice = 64 * M,
    .n_a_only_in_slice = 4 * M,
    .n_b_only_in_slice = 4 * M,
    .n_common_outside_before = 500 * M,
    .n_common_outside_after  = 500 * M,
    .n_a_only_outside_before = 100 * M,
    .n_a_only_outside_after  = 100 * M,
    .n_b_only_outside_before = 100 * M,
    .n_b_only_outside_after  = 100 * M,
  });

  add(Scenario{
    .name = "baseline_sparse",
    .slice_begin_ts = 50'000,
    .slice_end_ts = 0,
    .step_in_slice = 3 * M,
    .n_common_in_slice = 128 * M,
    .n_a_only_in_slice = 6 * M,
    .n_b_only_in_slice = 6 * M,
    .n_common_outside_before = 1000 * M,
    .n_common_outside_after  = 1000 * M,
    .n_a_only_outside_before = 200 * M,
    .n_a_only_outside_after  = 200 * M,
    .n_b_only_outside_before = 200 * M,
    .n_b_only_outside_after  = 200 * M,
  });

  for (int i = 1; i <= magnitude; ++i) {
    const std::uint64_t I = std::uint64_t(i);
    add(Scenario{
      .name = std::string("scale_dense_") + std::to_string(i),
      .slice_begin_ts = 200'000 + 10'000 * I,
      .slice_end_ts = 0,
      .step_in_slice = 1,
      .n_common_in_slice = 256 * I * I,
      .n_a_only_in_slice = 8 * I,
      .n_b_only_in_slice = 8 * I,
      .n_common_outside_before = 2000 * M,
      .n_common_outside_after  = 2000 * M,
      .n_a_only_outside_before = 400 * M,
      .n_a_only_outside_after  = 400 * M,
      .n_b_only_outside_before = 400 * M,
      .n_b_only_outside_after  = 400 * M,
    });

    add(Scenario{
      .name = std::string("scale_sparse_") + std::to_string(i),
      .slice_begin_ts = 500'000 + 10'000 * I,
      .slice_end_ts = 0,
      .step_in_slice = 2 * I,
      .n_common_in_slice = 512 * I,
      .n_a_only_in_slice = 16 * I,
      .n_b_only_in_slice = 16 * I,
      .n_common_outside_before = 3000 * M,
      .n_common_outside_after  = 3000 * M,
      .n_a_only_outside_before = 600 * M,
      .n_a_only_outside_after  = 600 * M,
      .n_b_only_outside_before = 600 * M,
      .n_b_only_outside_after  = 600 * M,
    });
  }

  add(Scenario{
    .name = std::string("stress_bigdiff_dyn_") + std::to_string(magnitude),
    .slice_begin_ts = 1'000'000,
    .slice_end_ts = 0,
    .step_in_slice = 1,
    // Heavier stress: large slice + large symmetric differences.
    // At magnitude=10 this yields: common=409,600; a_only=102,400; b_only=102,400 (slice only).
    .n_common_in_slice = 4096 * Q,
    .n_a_only_in_slice = 1024 * Q,
    .n_b_only_in_slice = 1024 * Q,
    // Outside-of-slice sizes also scale quadratically to keep branch/leaf pressure high.
    .n_common_outside_before = 2000 * Q,
    .n_common_outside_after  = 2000 * Q,
    .n_a_only_outside_before = 400 * Q,
    .n_a_only_outside_after  = 400 * Q,
    .n_b_only_outside_before = 400 * Q,
    .n_b_only_outside_after  = 400 * Q,
  });

  // Additional smaller stress points (kept from earlier versions)
  if (magnitude >= 2) {
    add(Scenario{
      .name = "stress_bigdiff_2",
      .slice_begin_ts = 900'000,
      .slice_end_ts = 0,
      .step_in_slice = 1,
      .n_common_in_slice = 1024 * 4,
      .n_a_only_in_slice = 64 * 2,
      .n_b_only_in_slice = 64 * 2,
      .n_common_outside_before = 4000 * 2,
      .n_common_outside_after  = 4000 * 2,
      .n_a_only_outside_before = 800 * 2,
      .n_a_only_outside_after  = 800 * 2,
      .n_b_only_outside_before = 800 * 2,
      .n_b_only_outside_after  = 800 * 2,
    });
  }

  if (magnitude >= 5) {
    add(Scenario{
      .name = "stress_bigdiff_5",
      .slice_begin_ts = 950'000,
      .slice_end_ts = 0,
      .step_in_slice = 1,
      .n_common_in_slice = 1024 * 25,
      .n_a_only_in_slice = 64 * 5,
      .n_b_only_in_slice = 64 * 5,
      .n_common_outside_before = 4000 * 5,
      .n_common_outside_after  = 4000 * 5,
      .n_a_only_outside_before = 800 * 5,
      .n_a_only_outside_after  = 800 * 5,
      .n_b_only_outside_before = 800 * 5,
      .n_b_only_outside_after  = 800 * 5,
    });
  }

  if (magnitude >= 8) {
    add(Scenario{
      .name = "stress_bigdiff_8",
      .slice_begin_ts = 980'000,
      .slice_end_ts = 0,
      .step_in_slice = 1,
      .n_common_in_slice = 1024 * 64,
      .n_a_only_in_slice = 64 * 8,
      .n_b_only_in_slice = 64 * 8,
      .n_common_outside_before = 4000 * 8,
      .n_common_outside_after  = 4000 * 8,
      .n_a_only_outside_before = 800 * 8,
      .n_a_only_outside_after  = 800 * 8,
      .n_b_only_outside_before = 800 * 8,
      .n_b_only_outside_after  = 800 * 8,
    });
  }

  // Sanity: ensure scenario names are unique (prepared-state directories use the name).

  {

    std::vector<std::string> names;

    names.reserve(out.size());

    for (const auto& s : out) names.push_back(s.name);

    std::sort(names.begin(), names.end());

    auto it = std::adjacent_find(names.begin(), names.end());

    if (it != names.end()) throw std::runtime_error(std::string("duplicate scenario name: ") + *it);

  }


  return out;
}

static inline Mode parse_mode(const std::string& s) {
  if (s == "init") return Mode::InitOnly;
  if (s == "bench") return Mode::BenchOnly;
  if (s == "both") return Mode::Both;
  throw std::runtime_error("bad --mode (use init|bench|both)");
}

static inline OutsideBeforeOrder parse_obefore(const std::string& s) {
  if (s == "desc") return OutsideBeforeOrder::Desc;
  if (s == "asc") return OutsideBeforeOrder::Asc;
  throw std::runtime_error("bad --outside-before-order (use asc|desc)");
}

static inline std::uint64_t parse_u64(const std::string& s) {
  std::uint64_t v = 0;
  for (char c : s) {
    if (c < '0' || c > '9') throw std::runtime_error("bad integer: " + s);
    v = v * 10 + std::uint64_t(c - '0');
  }
  return v;
}

static inline std::optional<Backend> parse_backend_opt(const std::string& s) {
  // --backend all | vector | aelmdb | btreelmdb
  // Returning nullopt means "all".
  if (s == "all") return std::nullopt;
  if (s == "vector") return Backend::Vector;
  if (s == "aelmdb" || s == "aelmdbslice" || s == "sliceaelmdb") return Backend::AELMDBSlice;
  if (s == "btreelmdb" || s == "btree" || s == "btreel") return Backend::BTreeLMDB;
  throw std::runtime_error("bad --backend (use all|vector|aelmdb|btreelmdb)");
}



#if defined(__unix__) || defined(__APPLE__)
static inline void spawn_bench_child(const char* argv0,
                                    int magnitude,
                                    const std::string& root,
                                    OutsideBeforeOrder ob_order,
                                    std::uint64_t commit_every,
                                    std::uint64_t mapsize_mb,
                                    std::uint64_t repeat_reconcile,
                                    const std::string& scenario,
                                    Backend backend) {
  std::vector<std::string> args;
  args.push_back(argv0);
  args.push_back("--mode"); args.push_back("bench");
  args.push_back("--magnitude"); args.push_back(std::to_string(magnitude));
  args.push_back("--root"); args.push_back(root);
  args.push_back("--outside-before-order"); args.push_back(obefore_order_name(ob_order));
  args.push_back("--commit-every"); args.push_back(std::to_string(commit_every));
  args.push_back("--mapsize-mb"); args.push_back(std::to_string(mapsize_mb));
  args.push_back("--repeat-reconcile"); args.push_back(std::to_string(repeat_reconcile));
  args.push_back("--scenario"); args.push_back(scenario);
  args.push_back("--backend");
  switch (backend) {
    case Backend::Vector: args.push_back("vector"); break;
    case Backend::AELMDBSlice: args.push_back("aelmdb"); break;
    case Backend::BTreeLMDB: args.push_back("btreelmdb"); break;
  }
  args.push_back("--no-header");

  std::vector<char*> cargs;
  cargs.reserve(args.size() + 1);
  for (auto& x : args) cargs.push_back(const_cast<char*>(x.c_str()));
  cargs.push_back(nullptr);

  pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("fork() failed");
  if (pid == 0) {
    execv(argv0, cargs.data());
    _exit(127);
  }

  int st = 0;
  if (waitpid(pid, &st, 0) < 0) throw std::runtime_error("waitpid() failed");
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
    throw std::runtime_error("isolated bench child failed for scenario='" + scenario + "' backend='" + backend_name(backend) + "'");
  }
}
#endif
} // namespace rbsr_adv

int main(int argc, char** argv) {
  using namespace rbsr_adv;

  int magnitude = 5;
  std::string root = "/tmp/rbsr_prepared";
  OutsideBeforeOrder ob_order = OutsideBeforeOrder::Desc;
  std::uint64_t commit_every = 0;
  std::uint64_t mapsize_mb = 1024;
  std::uint64_t repeat_reconcile = 1;
  bool isolate_bench = false;
  bool no_header = false;
  std::optional<std::string> only_scenario;
  Mode mode = Mode::Both;
  std::optional<Backend> only_backend; // nullopt => run all

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* opt)->std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + opt);
      return argv[++i];
    };

    if (a == "--magnitude") magnitude = int(parse_u64(need("--magnitude")));
    else if (a == "--root") root = need("--root");
    else if (a == "--outside-before-order") ob_order = parse_obefore(need("--outside-before-order"));
    else if (a == "--commit-every") commit_every = parse_u64(need("--commit-every"));
    else if (a == "--mapsize-mb") mapsize_mb = parse_u64(need("--mapsize-mb"));
    else if (a == "--repeat-reconcile") repeat_reconcile = parse_u64(need("--repeat-reconcile"));
    else if (a == "--isolate-bench") isolate_bench = true;
    else if (a == "--no-header") no_header = true;
    else if (a == "--scenario") only_scenario = need("--scenario");
    else if (a == "--mode") mode = parse_mode(need("--mode"));
    else if (a == "--backend") only_backend = parse_backend_opt(need("--backend"));
    else if (a == "--help" || a == "-h") {
      std::cout
        << "Usage: " << argv[0] << " [options]\n\n"
        << "Options:\n"
        << "  --magnitude N              Scenario magnitude (default 5)\n"
        << "  --root DIR                 Prepared-state root directory (default /tmp/rbsr_prepared)\n"
        << "  --outside-before-order asc|desc (default desc)\n"
        << "  --commit-every N           Preparation: commit every N inserts (0=single txn)\n"
        << "  --mapsize-mb N             LMDB map size in MiB (default 1024)\n"
        << "  --repeat-reconcile N       Bench: repeat reconciliation N times and average timings (default 1)\n"
        << "  --isolate-bench            Bench: fork+exec each row in a fresh process for trustworthy RSS\n"
        << "  --scenario NAME            Run only the named scenario (exact match)\n"
        << "  --no-header                Do not print CSV header (useful for isolate child runs)\n"
        << "  --backend all|vector|aelmdb|btreelmdb  Select backend(s) (default all)\n"
        << "  --mode init|bench|both     (default both)\n";
      return 0;
    }
    else {
      throw std::runtime_error("unknown arg: " + a);
    }
  }

  const auto scenarios_all = make_scenarios(magnitude);
  std::vector<Scenario> scenarios;
  scenarios.reserve(scenarios_all.size());
  if (only_scenario) {
    for (const auto& sc : scenarios_all) if (sc.name == *only_scenario) scenarios.push_back(sc);
    if (scenarios.empty()) throw std::runtime_error("--scenario not found: " + *only_scenario);
  } else {
    scenarios = scenarios_all;
  }

  if (!no_header) print_header_csv();

  for (const auto& sc : scenarios) {
    Expected exp = expected_for_slice(sc);

    // If benching from a prepared root, prefer the stored expected set.
    if (mode == Mode::BenchOnly || mode == Mode::Both) {
      if (auto oe = read_expected(root, sc)) exp = *oe;
    }

    auto run_one = [&](Backend b) {
      if (mode == Mode::InitOnly || mode == Mode::Both) {
        prepare_case(sc, b, ob_order, commit_every, mapsize_mb, root);
      }
      if (mode == Mode::BenchOnly || mode == Mode::Both) {
        if (isolate_bench) {
#if defined(__unix__) || defined(__APPLE__)
          spawn_bench_child(argv[0], magnitude, root, ob_order, commit_every, mapsize_mb, repeat_reconcile, sc.name, b);
#else
          throw std::runtime_error("--isolate-bench requires a POSIX platform (fork/exec)");
#endif
        } else {
          bench_case(sc, exp, magnitude, b, ob_order, commit_every, mapsize_mb, repeat_reconcile, root);
        }
      }
    };

    if (only_backend) {
      run_one(*only_backend);
    } else {
      run_one(Backend::Vector);
      run_one(Backend::BTreeLMDB);
      run_one(Backend::AELMDBSlice);
    }
  }

  return 0;
}
