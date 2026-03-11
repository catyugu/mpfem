/**
 * @file fe_cache.hpp
 * @brief Finite element cache for avoiding repeated FE creation
 * 
 * PERFORMANCE OPTIMIZATION:
 * Creating FE objects involves allocating and precomputing shape functions,
 * which is expensive. This cache reuses existing FE objects with the same
 * (GeometryType, degree, n_components) configuration.
 */

#ifndef MPFEM_FEM_FE_CACHE_HPP
#define MPFEM_FEM_FE_CACHE_HPP

#include "fe_base.hpp"
#include "core/exception.hpp"
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <functional>

namespace mpfem {

/**
 * @brief Key for FE cache lookup
 */
struct FEKey {
    GeometryType geom_type;
    int degree;
    int n_components;
    
    FEKey(GeometryType g, int d, int c) 
        : geom_type(g), degree(d), n_components(c) {}
    
    bool operator==(const FEKey& other) const {
        return geom_type == other.geom_type 
            && degree == other.degree 
            && n_components == other.n_components;
    }
};

/**
 * @brief Hash function for FEKey
 */
struct FEKeyHash {
    std::size_t operator()(const FEKey& k) const {
        return std::hash<int>{}(static_cast<int>(k.geom_type)) 
             ^ (std::hash<int>{}(k.degree) << 1)
             ^ (std::hash<int>{}(k.n_components) << 2);
    }
};

/**
 * @brief Thread-safe cache for finite element objects
 * 
 * USAGE:
 * @code
 * // Get the global cache
 * auto& cache = FECache::instance();
 * 
 * // Get or create FE (returns shared_ptr)
 * auto fe = cache.get(GeometryType::Tetrahedron, 1, 1);
 * 
 * // Use fe for assembly...
 * @endcode
 * 
 * PERFORMANCE:
 * - First call: creates and caches the FE
 * - Subsequent calls: returns cached FE (O(1) lookup)
 * - Thread-safe: uses mutex for concurrent access
 */
class FECache {
public:
    /// Get singleton instance
    static FECache& instance() {
        static FECache cache;
        return cache;
    }
    
    /**
     * @brief Get or create a finite element
     * @param geom_type Geometry type
     * @param degree Polynomial degree
     * @param n_components Number of components (1 for scalar, >1 for vector)
     * @return Shared pointer to the FE (cached, do not modify)
     */
    std::shared_ptr<const FiniteElement> get(
        GeometryType geom_type, int degree, int n_components = 1) {
        
        FEKey key(geom_type, degree, n_components);
        
        // Fast path: read-only lookup (no lock needed for read)
        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                return it->second;
            }
        }
        
        // Slow path: create and insert (needs exclusive lock)
        {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            // Double-check after acquiring exclusive lock
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                return it->second;
            }
            
            auto fe = create_fe(geom_type, degree, n_components);
            if (!fe) {
                MPFEM_THROW(InvalidArgument, 
                    "Failed to create FE for geometry type " 
                    << static_cast<int>(geom_type));
            }
            
            auto ptr = std::shared_ptr<const FiniteElement>(std::move(fe));
            cache_.emplace(key, ptr);
            return ptr;
        }
    }
    
    /**
     * @brief Check if a cached FE exists
     */
    bool contains(GeometryType geom_type, int degree, int n_components = 1) const {
        FEKey key(geom_type, degree, n_components);
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_.find(key) != cache_.end();
    }
    
    /**
     * @brief Get number of cached FEs
     */
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_.size();
    }
    
    /**
     * @brief Clear the cache
     */
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        cache_.clear();
    }
    
    // Non-copyable, non-movable
    FECache(const FECache&) = delete;
    FECache& operator=(const FECache&) = delete;
    
private:
    FECache() = default;
    
    mutable std::shared_mutex mutex_;
    std::unordered_map<FEKey, std::shared_ptr<const FiniteElement>, FEKeyHash> cache_;
};

/**
 * @brief Helper function to get cached FE
 * @param geom_type Geometry type
 * @param degree Polynomial degree  
 * @param n_components Number of components
 * @return Shared pointer to cached FE
 */
inline std::shared_ptr<const FiniteElement> get_cached_fe(
    GeometryType geom_type, int degree, int n_components = 1) {
    return FECache::instance().get(geom_type, degree, n_components);
}

}  // namespace mpfem

#endif  // MPFEM_FEM_FE_CACHE_HPP
