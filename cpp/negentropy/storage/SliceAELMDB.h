#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <algorithm>
#include <functional>
#include <string>
#include <string_view>

#include "lmdbxx/lmdb++.h"
#include "negentropy.h"

#ifndef MDB_AELMDB_VERSION
#error "LMDB header is not the AELMDB fork."
#endif

namespace negentropy { namespace storage {

/**
 * SliceAELMDB: Negentropy StorageBase backend over AELMDB aggregate APIs.
 *
 * This implementation is optimized for Negentropy's typical hot paths:
 *   - size() / getItem(i)
 *   - fingerprint(begin,end)
 *   - findLowerBound(begin,end,value)
 *   - iterate(begin,end,cb)
 *
 * It relies on AELMDB's aggregate/window extensions:
 *   - cursor.seek_rank(abs_rank)
 *   - dbi.window_fingerprint(..., MDB_agg_window& win, rel_begin, rel_end)
 *   - dbi.window_rank(..., MDB_agg_window& win, key)
 *
 * Required DBI schema (checked at runtime in the constructor):
 *   - MDB_AGG_ENTRIES (for rank/select and slice mapping)
 *   - MDB_AGG_HASHSUM (for fingerprints)
 *   - MDB_AGG_HASHSOURCE_FROM_KEY (so the hashsum matches the Item id bytes)
 *
 * Hash offset / key layout:
 *   For the (timestamp,id) schema used by Negentropy, we assume that:
 *     - the record key begins with an 8-byte big-endian timestamp
 *     - the Item id bytes begin at the per-DB hash offset (md_hash_offset)
 *     - MDB_HASH_SIZE == negentropy::ID_SIZE
 *
 * NOTE: If md_hash_offset > 8, the Bound-based constructor will fill the
 * bytes between the timestamp and the id with zeros. If your key schema uses
 * non-zero bytes there, use the raw-key constructor and pass full boundary
 * keys.
 */

static_assert(MDB_HASH_SIZE == negentropy::ID_SIZE,
              "SliceAELMDB expects MDB_HASH_SIZE == negentropy::ID_SIZE");

class AELMDBKeyCodecTSID {
public:
    static constexpr std::size_t TS_WIRE_SIZE = 8;
    static constexpr std::size_t ID_SIZE_     = negentropy::ID_SIZE;

    explicit AELMDBKeyCodecTSID(std::size_t id_offset)
        : id_offset_(id_offset) {
        if (id_offset_ < TS_WIRE_SIZE) {
            throw negentropy::err("SliceAELMDB: md_hash_offset < 8 (invalid for TSID keys)");
        }
    }

    std::size_t id_offset() const noexcept { return id_offset_; }
    std::size_t key_size() const noexcept { return id_offset_ + ID_SIZE_; }

    static uint64_t load_be64(const uint8_t* p) noexcept {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v = (v << 8) | uint64_t(p[i]);
        return v;
    }

    static void store_be64(uint8_t* p, uint64_t v) noexcept {
        for (int i = 7; i >= 0; --i) { p[i] = uint8_t(v & 0xffu); v >>= 8; }
    }

    /** Encode the minimum key for a Negentropy Bound.
     *
     * Produces a key that sorts at the beginning of the bound's timestamp
     * for the given id prefix length.
     */
    std::string encode_bound_min_key(const negentropy::Bound& b) const {
        std::string out(key_size(), '\0');
        store_be64(reinterpret_cast<uint8_t*>(out.data()), b.item.timestamp);
        // [8..id_offset_) left as zeros by construction.
        std::memcpy(out.data() + id_offset_,
                    b.item.id,
                    std::min<std::size_t>(b.idLen, ID_SIZE_));
        return out;
    }

    /** Validate and copy a raw key boundary.
     *
     * This is used when the caller already has an LMDB key (binary) and wants
     * to slice by that exact boundary.
     */
    std::string copy_key_bytes(std::string_view raw_key) const {
        if (raw_key.size() != key_size()) {
            throw negentropy::err("SliceAELMDB: bad raw key size (does not match DB schema)");
        }
        return std::string(raw_key.data(), raw_key.size());
    }

    /** Decode a Negentropy Item from an LMDB key. */
    negentropy::Item decode_item_from_key(std::string_view k) const {
        if (k.size() < key_size()) throw negentropy::err("SliceAELMDB: key too small");
        const auto* p = reinterpret_cast<const uint8_t*>(k.data());
        negentropy::Item it(load_be64(p));
        std::memcpy(it.id, p + id_offset_, ID_SIZE_);
        return it;
    }

private:
    std::size_t id_offset_;
};

struct SliceAELMDB : negentropy::StorageBase {
    lmdb::txn& txn;
    const lmdb::dbi& dbi;

    // Cached DBI schema.
    unsigned int agg_flags_ = 0;
    unsigned int hash_offset_ = 0; // md_hash_offset (bytes)

    // Key codec derived from the DBI's configured hash offset.
    AELMDBKeyCodecTSID codec_;

    // Optional [begin,end) boundaries for this slice, as raw LMDB keys.
    std::optional<std::string> begin_key_;
    std::optional<std::string> end_key_;

    // Cached string_view wrappers for begin/end keys, used by window_* APIs.
    std::optional<std::string_view> begin_sv_;
    std::optional<std::string_view> end_sv_;
    const std::string_view* lowp_{nullptr};
    const std::string_view* highp_{nullptr};

    // Reusable cursor (tied to txn lifetime) to avoid cursor_open/close overhead.
    mutable std::optional<lmdb::cursor> cur_;

    // Slice mapping in "absolute entry rank" space.
    uint64_t total_entries_ = 0;
    uint64_t abs_lo_ = 0;
    uint64_t abs_hi_ = 0;
    uint64_t slice_size_ = 0;

    // Cached slice mapping + fast fingerprints (combined window API).
    MDB_agg_window win_{}; // must be zero-initialized
    bool cached_full_fp_ = false;
    std::array<std::uint8_t, negentropy::ID_SIZE> full_hash_{};

    mutable negentropy::Item scratch_{};

    /**
     * Construct a slice from explicit raw boundary keys.
     *
     * begin_raw_key/end_raw_key are inclusive/exclusive in the sense of the
     * underlying AELMDB window API (the high boundary is treated as an upper
     * limit; see range_flags in init_abs_window_()).
     */
    SliceAELMDB(lmdb::txn& txn_,
                const lmdb::dbi& dbi_,
                std::optional<std::string_view> begin_raw_key = std::nullopt,
                std::optional<std::string_view> end_raw_key   = std::nullopt)
        : txn(txn_),
          dbi(dbi_),
          codec_(init_schema_and_get_id_offset_()) {

        if (begin_raw_key) begin_key_ = codec_.copy_key_bytes(*begin_raw_key);
        if (end_raw_key)   end_key_   = codec_.copy_key_bytes(*end_raw_key);
        cache_boundary_views_();
        init_abs_window_();
    }

    /** Construct a slice from Negentropy Bounds.
     *
     * This is the common path when slicing by timestamp ranges.
     */
    SliceAELMDB(lmdb::txn& txn_,
                const lmdb::dbi& dbi_,
                const negentropy::Bound& lower,
                const negentropy::Bound& upper)
        : txn(txn_),
          dbi(dbi_),
          codec_(init_schema_and_get_id_offset_()) {

        if (!(lower == negentropy::Bound(0))) {
            begin_key_ = codec_.encode_bound_min_key(lower);
        }
        if (!(upper == negentropy::Bound(negentropy::MAX_U64))) {
            end_key_ = codec_.encode_bound_min_key(upper);
        }
        cache_boundary_views_();
        init_abs_window_();
    }

    uint64_t size() override { return slice_size_; }

    const negentropy::Item& getItem(size_t i) override {
        if (i >= slice_size_) throw negentropy::err("SliceAELMDB: bad index");
        scratch_ = item_at_abs_rank_(abs_lo_ + uint64_t(i));
        return scratch_;
    }

    void iterate(size_t begin,
                 size_t end,
                 std::function<bool(const negentropy::Item&, size_t)> cb) override {
        check_bounds_(begin, end);
        if (begin == end) return;

        const uint64_t abs_begin = abs_lo_ + uint64_t(begin);
        const uint64_t abs_end   = abs_lo_ + uint64_t(end);

        auto& cur = cursor_();
        std::string_view k{}, v{};
        if (!cur.seek_rank(abs_begin, k, v))
            throw negentropy::err("SliceAELMDB: cursor seek_rank(abs_begin) failed");

        uint64_t abs_i = abs_begin;
        while (abs_i < abs_end) {
            scratch_ = codec_.decode_item_from_key(k);
            const size_t rel_i = size_t(abs_i - abs_lo_);

            if (!cb(scratch_, rel_i)) break;

            ++abs_i;
            if (abs_i >= abs_end) break;
            if (!cur.get(k, v, MDB_NEXT)) break;
        }
    }

    size_t findLowerBound(size_t begin,
                          size_t end,
                          const negentropy::Bound& value) override {
        check_bounds_(begin, end);

        const std::string bound_key = codec_.encode_bound_min_key(value);
        const std::string_view k{bound_key.data(), bound_key.size()};

        // Rank the key within the *current slice window*.
        const unsigned int range_flags = MDB_RANGE_LOWER_INCL;
        uint64_t rel = dbi.window_rank(txn,
                                       lowp_, nullptr,
                                       highp_, nullptr,
                                       range_flags,
                                       win_,
                                       k,
                                       nullptr);

        rel = std::min<uint64_t>(std::max<uint64_t>(rel, uint64_t(begin)), uint64_t(end));
        return size_t(rel);
    }

    negentropy::Fingerprint fingerprint(size_t begin, size_t end) override {
        check_bounds_(begin, end);

        const uint64_t n = uint64_t(end - begin);
        negentropy::Accumulator acc;
        acc.setToZero();

        if (n == 0) return acc.getFingerprint(0);

        // Fast path: Negentropy typically asks for the full-slice fingerprint first.
        if (begin == 0 && end == slice_size_ && cached_full_fp_) {
            std::memcpy(acc.buf, full_hash_.data(), negentropy::ID_SIZE);
            return acc.getFingerprint(n);
        }

        // Compute fingerprint directly in "rank space" within the precomputed window.
        const unsigned int range_flags = MDB_RANGE_LOWER_INCL;
        lmdb::agg a = dbi.window_fingerprint(txn,
                                             lowp_, nullptr,
                                             highp_, nullptr,
                                             range_flags,
                                             win_,
                                             uint64_t(begin),
                                             uint64_t(end));

        if (!a.has_hashsum() || !a.has_entries())
            throw negentropy::err("SliceAELMDB: DBI missing HASHSUM/ENTRIES");
        if (!a.has_hashsource_from_key())
            throw negentropy::err("SliceAELMDB: DBI must hash from key (MDB_AGG_HASHSOURCE_FROM_KEY)");

        if (a.mv_agg_entries != n)
            throw negentropy::err("SliceAELMDB: fingerprint range count mismatch");

        std::memcpy(acc.buf, a.hashsum_data(), negentropy::ID_SIZE);
        return acc.getFingerprint(n);
    }

private:
    void check_bounds_(size_t begin, size_t end) const {
        if (begin > end || end > slice_size_)
            throw negentropy::err("SliceAELMDB: bad range");
    }

    lmdb::cursor& cursor_() const {
        if (!cur_) cur_.emplace(lmdb::cursor::open(txn, dbi));
        return *cur_;
    }

    void cache_boundary_views_() {
        if (begin_key_) {
            begin_sv_ = std::string_view(begin_key_->data(), begin_key_->size());
            lowp_ = &*begin_sv_;
        }
        if (end_key_) {
            end_sv_ = std::string_view(end_key_->data(), end_key_->size());
            highp_ = &*end_sv_;
        }
    }

    std::size_t init_schema_and_get_id_offset_() {
#if !defined(MDB_AGG_MASK)
        throw negentropy::err("SliceAELMDB requires AELMDB aggregate support (MDB_AGG_MASK)");
#else
        agg_flags_ = dbi.agg_flags(txn);
        if ((agg_flags_ & MDB_AGG_ENTRIES) == 0)
            throw negentropy::err("SliceAELMDB: DBI missing MDB_AGG_ENTRIES");
        if ((agg_flags_ & MDB_AGG_HASHSUM) == 0)
            throw negentropy::err("SliceAELMDB: DBI missing MDB_AGG_HASHSUM");
#ifdef MDB_AGG_HASHSOURCE_FROM_KEY
        if ((agg_flags_ & MDB_AGG_HASHSOURCE_FROM_KEY) == 0)
            throw negentropy::err("SliceAELMDB: DBI missing MDB_AGG_HASHSOURCE_FROM_KEY");
#endif

        // Fetch per-DB md_hash_offset via the AELMDB extension API.
        // This is part of the persistent MDB_db header and is configured with mdb_set_hash_offset().
        unsigned int off = 0;
        const int grc = ::mdb_get_hash_offset(txn, dbi.handle(), &off);
        if (grc != MDB_SUCCESS)
            throw negentropy::err("SliceAELMDB: mdb_get_hash_offset failed");
        hash_offset_ = off;
        return std::size_t(hash_offset_);
#endif
    }

    void init_abs_window_() {
        // Initialize the window once using the combined API. This computes:
        // - total_entries_ (total records in DB)
        // - abs_lo_/abs_hi_ for the slice [begin_key_, end_key_)
        // - (optionally) the full-slice fingerprint, which we cache

        const unsigned int range_flags = MDB_RANGE_LOWER_INCL;
        lmdb::agg full = dbi.window_fingerprint(txn,
                                                lowp_, nullptr,
                                                highp_, nullptr,
                                                range_flags,
                                                win_,
                                                0,
                                                MDB_AGG_WINDOW_END);

        if (!full.has_entries())
            throw negentropy::err("SliceAELMDB: DBI missing MDB_AGG_ENTRIES");

        total_entries_ = win_.mv_total_entries;
        abs_lo_ = win_.mv_abs_lo;
        abs_hi_ = win_.mv_abs_hi;

        if (abs_hi_ < abs_lo_) abs_hi_ = abs_lo_;
        slice_size_ = abs_hi_ - abs_lo_;

        if (full.has_hashsum() && full.mv_agg_entries == slice_size_) {
            std::memcpy(full_hash_.data(), full.hashsum_data(), negentropy::ID_SIZE);
            cached_full_fp_ = true;
        }
    }

    negentropy::Item item_at_abs_rank_(uint64_t abs_rank) const {
        auto& cur = cursor_();
        std::string_view k{}, v{};
        if (!cur.seek_rank(abs_rank, k, v))
            throw negentropy::err("SliceAELMDB: seek_rank(abs_rank) out of range");
        return codec_.decode_item_from_key(k);
    }
};

struct WholeAELMDB : SliceAELMDB {
    WholeAELMDB(lmdb::txn& txn, const lmdb::dbi& dbi)
        : SliceAELMDB(txn, dbi, std::nullopt, std::nullopt) {}
};

}} // namespace negentropy::storage
