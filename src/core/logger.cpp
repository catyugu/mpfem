#include "logger.hpp"
#include <iomanip>
#include <iostream>

namespace mpfem {

// =============================================================================
// Logger implementation
// =============================================================================

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::setLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(instance().mutex_);
    instance().minimumLevel_ = level;
}

void Logger::log(LogLevel level, const std::string& message) {
    Logger& logger = instance();
    
    // Check level threshold
    if (static_cast<int>(level) < static_cast<int>(logger.minimumLevel_)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(logger.mutex_);
    
    // Format: [LEVEL] [elapsed] message
    const char* levelStr = "";
    switch (level) {
        case LogLevel::Debug:   levelStr = "DEBUG"; break;
        case LogLevel::Info:    levelStr = "INFO"; break;
        case LogLevel::Warning: levelStr = "WARN"; break;
        case LogLevel::Error:   levelStr = "ERROR"; break;
    }
    
    std::cout << "[" << levelStr << "] "
              << "[" << formatElapsed() << "] "
              << message << std::endl;
}

std::chrono::milliseconds Logger::elapsedMillis() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        now - instance().startTime_);
}

std::string Logger::formatElapsed() {
    auto ms = elapsedMillis().count();
    
    std::ostringstream oss;
    if (ms < 1000) {
        oss << ms << "ms";
    } else if (ms < 60000) {
        oss << std::fixed << std::setprecision(2) << (ms / 1000.0) << "s";
    } else {
        int seconds = static_cast<int>(ms / 1000);
        int minutes = seconds / 60;
        seconds %= 60;
        oss << minutes << "m" << seconds << "s";
    }
    
    return oss.str();
}

// =============================================================================
// ScopedTimer implementation
// =============================================================================

ScopedTimer::ScopedTimer(const std::string& label, LogLevel level)
    : label_(label)
    , level_(level)
    , start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    if (!stopped_) {
        stop();
    }
}

void ScopedTimer::stop() {
    if (!stopped_) {
        end_ = std::chrono::steady_clock::now();
        stopped_ = true;
        
        Logger::log(level_, label_ + " completed in " + getElapsedStr());
    }
}

double ScopedTimer::getElapsedSeconds() const {
    auto end = stopped_ ? end_ : std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start_);
    return duration.count();
}

std::string ScopedTimer::getElapsedStr() const {
    double seconds = getElapsedSeconds();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << seconds << "s";
    return oss.str();
}

}  // namespace mpfem
