#ifndef MPFEM_LOGGER_HPP
#define MPFEM_LOGGER_HPP

#include <chrono>
#include <mutex>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>

namespace mpfem {

/**
 * @brief Logging severity levels used by Logger.
 */
enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error
};

/**
 * @brief Minimal thread-safe logger used by mpfem modules.
 * 
 * Singleton pattern for global access. Supports:
 * - Log level filtering
 * - Thread-safe output
 * - Elapsed time tracking
 * 
 * Usage:
 *   Logger::setLevel(LogLevel::Debug);
 *   Logger::log(LogLevel::Info, "Starting computation");
 *   LOG_INFO << "Processing element " << elemId;
 *   LOG_ERROR << "Error in element " << elemId << ": " << errorMsg;
 */
class Logger {
public:
    /**
     * @brief Sets minimum severity threshold for output.
     */
    static void setLevel(LogLevel level);

    /**
     * @brief Logs a message when level is above configured threshold.
     */
    static void log(LogLevel level, const std::string& message);

    /**
     * @brief Returns elapsed milliseconds since program start.
     */
    static std::chrono::milliseconds elapsedMillis();

    /**
     * @brief Formats elapsed time as human-readable string.
     */
    static std::string formatElapsed();

    /**
     * @brief Get the singleton instance.
     */
    static Logger& instance();

private:
    Logger() = default;
    
    std::mutex mutex_;
    LogLevel minimumLevel_ = LogLevel::Info;
    std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now();
};

/**
 * @brief RAII timer that logs elapsed time on destruction.
 * 
 * Usage:
 *   {
 *       ScopedTimer timer("Assembly");
 *       // ... do work ...
 *   }  // Logs: "Assembly completed in 1.23s"
 */
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& label, LogLevel level = LogLevel::Info);
    ~ScopedTimer();

    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

    void stop();
    double getElapsedSeconds() const;
    std::string getElapsedStr() const;

private:
    std::string label_;
    LogLevel level_;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point end_;
    bool stopped_ = false;
};

// =============================================================================
// Stream-style logging helper class
// =============================================================================

/**
 * @brief Temporary object for stream-style logging.
 * 
 * Usage: LOG_INFO << "Value: " << value;
 * The log message is sent on destruction.
 */
class LogMessage {
public:
    explicit LogMessage(LogLevel level) : level_(level) {}
    
    // Non-copyable
    LogMessage(const LogMessage&) = delete;
    LogMessage& operator=(const LogMessage&) = delete;
    
    // Movable (allows LOG_INFO << ... to work)
    LogMessage(LogMessage&& other) noexcept 
        : level_(other.level_), oss_(std::move(other.oss_)) {}
    
    ~LogMessage() {
        if (!oss_.str().empty() || level_ == LogLevel::Error) {
            Logger::instance().log(level_, oss_.str());
        }
    }
    
    template<typename T>
    LogMessage& operator<<(const T& value) {
        oss_ << value;
        return *this;
    }
    
    // Support std::endl and other manipulators
    LogMessage& operator<<(std::ostream& (*manip)(std::ostream&)) {
        oss_ << manip;
        return *this;
    }
    
private:
    LogLevel level_;
    std::ostringstream oss_;
};

// =============================================================================
// Convenience macros for logging (stream-style)
// =============================================================================

#define LOG_DEBUG   ::mpfem::LogMessage(::mpfem::LogLevel::Debug)
#define LOG_INFO    ::mpfem::LogMessage(::mpfem::LogLevel::Info)
#define LOG_WARNING    ::mpfem::LogMessage(::mpfem::LogLevel::Warning)
#define LOG_ERROR   ::mpfem::LogMessage(::mpfem::LogLevel::Error)

/**
 * @brief Assertion macro that logs error and exits on failure.
 * 
 * Usage: Check(condition, "Error message");
 * If condition is false, logs error and calls std::exit(1).
 */
#define Check(cond, msg)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            ::mpfem::LogMessage(::mpfem::LogLevel::Error) << msg; \
            std::exit(1);                                       \
        }                                                       \
    } while (0)

}  // namespace mpfem

#endif  // MPFEM_LOGGER_HPP
