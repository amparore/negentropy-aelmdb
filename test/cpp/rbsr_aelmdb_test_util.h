#pragma once

// Test utilities for negentropy + AELMDB-backed storage (SliceAELMDB).
//
// Schema used by these tests:
//   key = (timestamp_be64 || id_bytes[32])
//   value = empty
//
// We enable AELMDB aggregate metadata so SliceAELMDB can answer negentropy queries
// via rank/select + range fingerprints instead of scanning.

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "lmdbxx/lmdb++.h"
#include "lmdb.h"

#ifndef MDB_AELMDB_VERSION
#error "This project requires the AELMDB fork of LMDB: lmdb.h must define MDB_AELMDB_VERSION."
#endif

namespace rbsr_test {

// Key layout used throughout the tests.
//
// We deliberately keep the test schema simple (timestamp,id) so it's easy to
// reason about correctness. In the current SliceAELMDB implementation, the
// timestamp is an 8-byte big-endian prefix at offset 0.
static constexpr std::size_t TS_OFFSET    = 0;  // where timestamp_be64 lives
static constexpr std::size_t TS_WIRE_SIZE = 8;  // bytes
static constexpr std::size_t ID_SIZE  = 32;  // id payload returned by negentropy
static constexpr std::size_t HASH_OFFSET = TS_OFFSET + TS_WIRE_SIZE; // where id bytes live
static constexpr std::size_t KEY_SIZE = HASH_OFFSET + ID_SIZE;

using KeyBytes = std::array<std::uint8_t, KEY_SIZE>;
using IdBytes  = std::array<std::uint8_t, ID_SIZE>;

// NOTE: This test targets the current SliceAELMDB implementation in this repo.
// SliceAELMDB derives its key layout from the DBI:
//   - timestamp is an 8-byte big-endian prefix at offset 0
//   - id bytes start at md_hash_offset (configured via mdb_set_hash_offset)
// Therefore, the only key-layout knob in these tests is HASH_OFFSET (== 8).

inline void store_be64(std::uint8_t* p, std::uint64_t v) noexcept {
    for (int i = 7; i >= 0; --i) {
        p[i] = static_cast<std::uint8_t>(v & 0xffu);
        v >>= 8;
    }
}

inline KeyBytes make_boundary_key(std::uint64_t ts) {
    // Boundary key uses an all-zero id so it compares before any real id for the same ts.
    KeyBytes k{};
    store_be64(k.data() + TS_OFFSET, ts);
    std::memset(k.data() + HASH_OFFSET, 0, ID_SIZE);
    return k;
}

inline IdBytes pack_id_le(std::uint64_t n) {
    // Produce a deterministic 32-byte id from a u64.
    // Only the first 8 bytes are used; remainder stays zero.
    IdBytes id{};
    for (int i = 0; i < 8; ++i) {
        id[i] = static_cast<std::uint8_t>((n >> (8 * i)) & 0xffu);
    }
    return id;
}

inline std::uint64_t unpack_id_le(std::string_view id) {
    if (id.size() != ID_SIZE) throw std::runtime_error("unpack_id_le: bad id size");
    std::uint64_t n = 0;
    for (int i = 0; i < 8; ++i) {
        n |= (std::uint64_t(static_cast<std::uint8_t>(id[i])) << (8 * i));
    }
    return n;
}

inline KeyBytes make_key(std::uint64_t ts, const IdBytes& id) {
    KeyBytes k{};
    store_be64(k.data() + TS_OFFSET, ts);
    std::memcpy(k.data() + HASH_OFFSET, id.data(), ID_SIZE);
    return k;
}

// AELMDB-backed peer DB: one item == one LMDB key, and aggregates are enabled.
struct PeerAELMDB {
    std::string path;
    lmdb::env env;
    lmdb::dbi dbi;

    static PeerAELMDB create_fresh(const std::string& dir, const char* dbname,
                                  unsigned hash_offset = static_cast<unsigned>(HASH_OFFSET)) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir);

        auto env = lmdb::env::create();
        env.set_max_dbs(16);
        env.set_mapsize(256ull * 1024 * 1024);
        env.open(dir.c_str(), 0);

        lmdb::dbi dbi{0};
        {
            auto txn = lmdb::txn::begin(env);

            // Enable AELMDB aggregates used by SliceAELMDB:
            // - entries counts for rank/select
            // - hashsum for range fingerprinting
            // - hashsource-from-key because our "hash material" is embedded in the key
            const unsigned int dbi_flags =
                MDB_CREATE |
                MDB_AGG_ENTRIES |
                MDB_AGG_HASHSUM |
                MDB_AGG_HASHSOURCE_FROM_KEY;

            dbi = lmdb::dbi::open(txn, dbname, dbi_flags);

            // Important: SliceAELMDB reads this from the DBI (via mdb_get_hash_offset()).
            // In our key schema, the id begins immediately after the timestamp prefix.
            dbi.set_hash_offset(txn, hash_offset);

            txn.commit();
        }

        return PeerAELMDB{dir, std::move(env), dbi};
    }
};

// Plain LMDB env helper (used by tests that build non-AELMDB backends inside an env).
struct FreshEnv {
    std::string path;
    lmdb::env env;

    static FreshEnv create_fresh(const std::string& dir) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir);

        auto env = lmdb::env::create();
        env.set_max_dbs(16);
        env.set_mapsize(256ull * 1024 * 1024);
        env.open(dir.c_str(), 0);

        return FreshEnv{dir, std::move(env)};
    }
};

inline void insert_item(lmdb::txn& txn, const lmdb::dbi& dbi, std::uint64_t ts, const IdBytes& id) {
    const KeyBytes kb = make_key(ts, id);
    const std::string_view k(reinterpret_cast<const char*>(kb.data()), kb.size());
    const std::string_view v{};

    const bool ok = const_cast<lmdb::dbi&>(dbi).put(txn, k, v, MDB_NOOVERWRITE);
    if (!ok) throw std::runtime_error("insert_item: duplicate key (MDB_NOOVERWRITE)");
}

inline void require(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

inline std::vector<std::uint64_t> ids_to_sorted_u64(const std::vector<std::string>& ids) {
    std::vector<std::uint64_t> out;
    out.reserve(ids.size());
    for (const auto& s : ids) out.push_back(unpack_id_le(s));
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace rbsr_test
