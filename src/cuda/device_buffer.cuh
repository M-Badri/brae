#pragma once
// cf GPU offload foundation (G0): DeviceBuffer<T>, RAII device memory with explicit H2D/D2H. Explicit
// device buffers (not managed memory) keep the residency discipline visible: every host<->device copy is
// a named call, so the G7 goal (device-resident SIMPLE loop, zero copies per iteration) is measurable.
// The CPU baseline stays the oracle; each GPU kernel is validated against it.
#include "cf_types.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>          // BRAE_POOL_STATS: allocation-size histogram
#include <algorithm>    // sort, for the histogram report
#include <unordered_map>
#include <cstdlib>
#include <cstdio>

namespace brae {

inline void cudaCheck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string("brae cuda: ") + what + ": " + cudaGetErrorString(e));
}

// Entry-count census for the caches keyed on a field pointer, under BRAE_CACHE_STATS=1. Such a cache
// stays bounded only while the pointer is stable: the legacy driver allocates its psi fresh every outer
// iteration, so a cache keyed on it gains an entry per solve (items 75, 76). Prints on each new high mark.
inline void cacheStat(const char* name, std::size_t n)
{
    static const bool on = std::getenv("BRAE_CACHE_STATS") != nullptr;
    if (!on) return;
    static std::unordered_map<std::string, std::size_t> high;
    std::size_t& h = high[name];
    if (n > h) { h = n; std::fprintf(stderr, "[cache] %s entries=%zu\n", name, n); }
}

namespace detail {
// Caching device allocator: blocks freed by a DeviceBuffer are RETAINED in a size-keyed free list and handed back
// to the next same-size request, instead of round-tripping through cudaMalloc/cudaFree on every temporary. The
// device SIMPLE loop churns ~290 malloc+free/iter on per-iteration temporaries (nsys: cudaMalloc+cudaFree = 36.5%
// of host API time after the sync work); once every size is warm (after iter 1) steady-state malloc/free -> ~0.
// Returned memory is NOT zeroed, same contract as cudaMalloc, and cf always writes a buffer before reading it.
// Single-threaded use (cf runs one solve on one host thread; the GPU path is not entered concurrently). Disable
// with BRAE_NO_DEVICE_POOL=1 to A/B against the raw-cudaMalloc behaviour. The pool is intentionally leaked (never
// destructed) so there is no static-destruction-order hazard and no cudaFree after the CUDA context is torn down.
class DevicePool
{
public:
    DevicePool() : enabled_(std::getenv("BRAE_NO_DEVICE_POOL") == nullptr) {}
    void* take(std::size_t bytes)
    {
        if (bytes == 0) return nullptr;
        if (enabled_)
        {
            auto it = free_.find(bytes);
            if (it != free_.end() && !it->second.empty())
            {
                void* p = it->second.back();
                it->second.pop_back();
                heldBytes_ -= bytes;
                return p;
            }
        }
        void* p = nullptr;
        cudaCheck(cudaMalloc(&p, bytes), "pool cudaMalloc");
        mallocBytes_ += bytes; ++mallocs_;
        hist_[bytes].first += bytes; ++hist_[bytes].second;
        return p;
    }
    void give(void* p, std::size_t bytes)
    {
        if (!p) return;
        if (enabled_ && bytes) { free_[bytes].push_back(p); heldBytes_ += bytes; }   // retained, never freed
        else cudaFree(p);
    }
    // BRAE_POOL_STATS=1: what the pool asked the driver for, and how much of that is sitting in the
    // free list rather than in use. The list is keyed on EXACT size, so a block is reusable only by a
    // request of exactly the same byte count -- a run that asks for many distinct sizes retains them all.
    void report(const char* when) const
    {
        if (!std::getenv("BRAE_POOL_STATS")) return;
        std::size_t sizes = free_.size(), blocks = 0;
        for (const auto& kv : free_) blocks += kv.second.size();
        std::fprintf(stderr,
            "[pool] %-10s cudaMalloc %8.3f GB in %zu calls | idle in free list %8.3f GB "
            "(%zu distinct sizes, %zu blocks)\n",
            when, mallocBytes_/1073741824.0, mallocs_, heldBytes_/1073741824.0, sizes, blocks);
        // The biggest consumers, by total bytes. Sizes are the attribution: a buffer is sized by what it
        // spans, so nCells*8 is a cell field, nFaces*8 a face field, and the tail of odd sizes is the
        // AMG hierarchy's coarse levels. Cheaper and more honest than guessing at call sites.
        std::vector<std::pair<std::size_t, std::pair<std::size_t, std::size_t>>> v(hist_.begin(), hist_.end());
        std::sort(v.begin(), v.end(),
                  [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
        const std::size_t n = v.size() < 12 ? v.size() : 12;
        for (std::size_t i = 0; i < n; ++i)
        {
            std::fprintf(stderr, "[pool]   %10.2f MB total  %5zu x %10.2f MB\n",
                         v[i].second.first/1048576.0, v[i].second.second, v[i].first/1048576.0);
        }
    }
private:
    std::unordered_map<std::size_t, std::vector<void*>> free_;
    std::size_t mallocBytes_ = 0, heldBytes_ = 0, mallocs_ = 0;
    std::map<std::size_t, std::pair<std::size_t, std::size_t>> hist_;   // size -> {total bytes, count}
    bool enabled_;
};
inline DevicePool& devicePool() { static DevicePool* p = new DevicePool(); return *p; }   // intentionally leaked
} // namespace detail

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { resize(n); }
    explicit DeviceBuffer(const std::vector<T>& h) { resize(h.size()); copyFrom(h); }
    ~DeviceBuffer() { if (d_) detail::devicePool().give(d_, n_ * sizeof(T)); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : d_(o.d_), n_(o.n_) { o.d_ = nullptr; o.n_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept
    {
        if (this != &o)
        {
            if (d_) detail::devicePool().give(d_, n_ * sizeof(T));
            d_ = o.d_;
            n_ = o.n_;
            o.d_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    void resize(std::size_t n)
    {
        if (n == n_) return;
        if (d_) detail::devicePool().give(d_, n_ * sizeof(T));            // return OLD block (OLD n_) to the pool
        n_ = n;
        d_ = nullptr;
        if (n_) d_ = static_cast<T*>(detail::devicePool().take(n_ * sizeof(T)));
    }
    void copyFrom(const std::vector<T>& h)                                // H2D
    {
        if (h.size() != n_) resize(h.size());
        cudaCheck(cudaMemcpy(d_, h.data(), n_ * sizeof(T), cudaMemcpyHostToDevice), "H2D");
    }
    void copyTo(std::vector<T>& h) const                                 // D2H
    {
        h.resize(n_);
        cudaCheck(cudaMemcpy(h.data(), d_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
    }
    std::vector<T> host() const { std::vector<T> h; copyTo(h); return h; }

    T*       data()       { return d_; }
    const T* data() const { return d_; }
    std::size_t size() const { return n_; }

private:
    T*          d_ = nullptr;
    std::size_t n_ = 0;
};

} // namespace brae
