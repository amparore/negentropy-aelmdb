#pragma once

#include <array>
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

static constexpr std::size_t TS_SIZE  = 8;
static constexpr std::size_t ID_SIZE  = 32;
static constexpr std::size_t KEY_SIZE = TS_SIZE + ID_SIZE;

using KeyBytes = std::array<std::uint8_t, KEY_SIZE>;
using IdBytes  = std::array<std::uint8_t, ID_SIZE>;

inline void store_be64(std::uint8_t* p, std::uint64_t v) noexcept {
    for (int i = 7; i >= 0; --i) { p[i] = std::uint8_t(v & 0xffu); v >>= 8; }
}

inline std::uint64_t load_be64(const std::uint8_t* p) noexcept {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | std::uint64_t(p[i]);
    return v;
}

inline KeyBytes make_boundary_key(std::uint64_t ts) {
    KeyBytes k{};
    store_be64(k.data(), ts);
    std::memset(k.data() + TS_SIZE, 0, ID_SIZE);
    return k;
}

inline IdBytes pack_id_le(std::uint64_t n) {
    IdBytes id{};
    for (int i = 0; i < 8; ++i) id[i] = std::uint8_t((n >> (8 * i)) & 0xffu);
    return id;
}

inline std::uint64_t unpack_id_le(std::string_view id) {
    if (id.size() != ID_SIZE) throw std::runtime_error("unpack_id_le: bad id size");
    std::uint64_t n = 0;
    for (int i = 0; i < 8; ++i) n |= (std::uint64_t(std::uint8_t(id[i])) << (8 * i));
    return n;
}

inline KeyBytes make_key(std::uint64_t ts, const IdBytes& id) {
    KeyBytes k{};
    store_be64(k.data(), ts);
    std::memcpy(k.data() + TS_SIZE, id.data(), ID_SIZE);
    return k;
}

struct PeerDB {
    std::string path;
    lmdb::env env;
    lmdb::dbi dbi;

    // Create a fresh AELMDB database with aggregates enabled.
    // mapsize_mb must be large enough for the chosen scenario magnitude.
    static PeerDB create_fresh(const std::string& dir, const char* dbname, std::uint64_t mapsize_mb) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir);

        auto env = lmdb::env::create();
        env.set_max_dbs(16);
        env.set_mapsize(std::uint64_t(mapsize_mb) * 1024ull * 1024ull);
        env.open(dir.c_str(), 0);

        lmdb::dbi dbi{0};

        {
            auto txn = lmdb::txn::begin(env);

            const unsigned int dbi_flags =
                MDB_CREATE |
                MDB_AGG_ENTRIES |
                MDB_AGG_HASHSUM |
                MDB_AGG_HASHSOURCE_FROM_KEY;

            dbi = lmdb::dbi::open(txn, dbname, dbi_flags);
            // TSID keys: timestamp is 8 bytes at offset 0, id begins at offset 8.
            dbi.set_hash_offset(txn, /*hash_offset=*/8);

            txn.commit();
        }

        return PeerDB{dir, std::move(env), dbi};
    }
};

inline void insert_item(lmdb::txn& txn, const lmdb::dbi& dbi, std::uint64_t ts, const IdBytes& id,
                       unsigned put_flags = MDB_NOOVERWRITE) {
    const KeyBytes kb = make_key(ts, id);
    const std::string_view k(reinterpret_cast<const char*>(kb.data()), kb.size());
    const std::string_view v{};
    const bool ok = const_cast<lmdb::dbi&>(dbi).put(txn, k, v, put_flags);
    if (!ok) throw std::runtime_error("insert_item: duplicate key (MDB_NOOVERWRITE)");
}

inline void require(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

} // namespace rbsr_test
