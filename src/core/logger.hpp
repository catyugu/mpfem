/**
 * @file logger.hpp
 * @brief Minimal thread-safe logging system for mpfem
 * 
 * Provides three log levels: INFO, WARN, ERROR.
 * Log level filtering is done at compile time.
 */

#ifndef MPFEM_CORE_LOGGER_HPP
#define MPFEM_CORE_LOGGER_HPP

#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>

namespace mpfem {

// ============================================================
// Log Level Definition
// ============================================================

enum class LogLevel : int {
    INFO = 0,
    WARN = 1,
    ERROR = 2,
    NONE = 99  // Disable all logging
};

// Compile-time log level (set via compile definition)
#ifndef MPFEM_LOG_LEVEL
#define MPFEM_LOG_LEVEL LogLevel::INFO
#endif

// ============================================================
// Logger Class (Thread-Safe Singleton)
// ============================================================

class Logger {
public:
    /// Get singleton instance
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    /// Set minimum log level at runtime
    void set_level(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        min_level_ = level;
    }

    /// Log a message at the specified level
    void log(LogLevel level, std::string_view file, int line,
             std::string_view message) {
        if (static_cast<int>(level) < static_cast<int>(min_level_)) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Get current time
        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch()) %
                      1000;

        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &now_time);
#else
        localtime_r(&now_time, &tm_buf);
#endif

        // Format: [LEVEL] YYYY-MM-DD HH:MM:SS.mmm [file:line] message
        std::ostringstream oss;
        oss << '[' << level_str(level) << "] ";
        oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
        oss << " [" << file << ':' << line << "] ";
        oss << message << '\n';

        // Output to stderr for ERROR, stdout for others
        if (level == LogLevel::ERROR) {
            std::cerr << oss.str();
        } else {
            std::cout << oss.str();
        }
    }

    /// Log without file/line info
    void log(LogLevel level, std::string_view message) {
        if (static_cast<int>(level) < static_cast<int>(min_level_)) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch()) %
                      1000;

        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &now_time);
#else
        localtime_r(&now_time, &tm_buf);
#endif

        std::ostringstream oss;
        oss << '[' << level_str(level) << "] ";
        oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
        oss << " " << message << '\n';

        if (level == LogLevel::ERROR) {
            std::cerr << oss.str();
        } else {
            std::cout << oss.str();
        }
    }

private:
    Logger() : min_level_(MPFEM_LOG_LEVEL) {}
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    const char* level_str(LogLevel level) const {
        switch (level) {
            case LogLevel::INFO:
                return "INFO";
            case LogLevel::WARN:
                return "WARN";
            case LogLevel::ERROR:
                return "ERROR";
            default:
                return "UNKNOWN";
        }
    }

    std::mutex mutex_;
    LogLevel min_level_;
};

// ============================================================
// Log Macros
// ============================================================

// Helper to extract filename from path
inline std::string_view extract_filename(std::string_view path) {
    auto pos = path.find_last_of("/\\");
    return (pos == std::string_view::npos) ? path : path.substr(pos + 1);
}

// Main logging macros
#define MPFEM_INFO(msg)                                            \
    do {                                                           \
        if (static_cast<int>(mpfem::LogLevel::INFO) >=             \
            static_cast<int>(MPFEM_LOG_LEVEL)) {                   \
            std::ostringstream _oss;                               \
            _oss << msg;                                           \
            mpfem::Logger::instance().log(                         \
                mpfem::LogLevel::INFO,                             \
                mpfem::extract_filename(__FILE__), __LINE__,       \
                _oss.str());                                       \
        }                                                          \
    } while (0)

#define MPFEM_WARN(msg)                                            \
    do {                                                           \
        if (static_cast<int>(mpfem::LogLevel::WARN) >=             \
            static_cast<int>(MPFEM_LOG_LEVEL)) {                   \
            std::ostringstream _oss;                               \
            _oss << msg;                                           \
            mpfem::Logger::instance().log(                         \
                mpfem::LogLevel::WARN,                             \
                mpfem::extract_filename(__FILE__), __LINE__,       \
                _oss.str());                                       \
        }                                                          \
    } while (0)

#define MPFEM_ERROR(msg)                                           \
    do {                                                           \
        if (static_cast<int>(mpfem::LogLevel::ERROR) >=            \
            static_cast<int>(MPFEM_LOG_LEVEL)) {                   \
            std::ostringstream _oss;                               \
            _oss << msg;                                           \
            mpfem::Logger::instance().log(                         \
                mpfem::LogLevel::ERROR,                            \
                mpfem::extract_filename(__FILE__), __LINE__,       \
                _oss.str());                                       \
        }                                                          \
    } while (0)

// Conditional logging (only logs if condition is true)
#define MPFEM_INFO_IF(cond, msg)                                   \
    do {                                                           \
        if (cond) {                                                \
            MPFEM_INFO(msg);                                       \
        }                                                          \
    } while (0)

#define MPFEM_WARN_IF(cond, msg)                                   \
    do {                                                           \
        if (cond) {                                                \
            MPFEM_WARN(msg);                                       \
        }                                                          \
    } while (0)

#define MPFEM_ERROR_IF(cond, msg)                                  \
    do {                                                           \
        if (cond) {                                                \
            MPFEM_ERROR(msg);                                      \
        }                                                          \
    } while (0)

}  // namespace mpfem

#endif  // MPFEM_CORE_LOGGER_HPP
