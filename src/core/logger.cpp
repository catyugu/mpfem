#include "logger.hpp"
#include <iomanip>
#include <iostream>

namespace mpfem {

    LogMessage::LogMessage(LogLevel level)
        : level_(level)
    {
    }

    LogMessage::LogMessage(LogMessage&& other) noexcept
        : level_(other.level_)
        , buffer_(std::move(other.buffer_))
    {
    }

    LogMessage::~LogMessage()
    {
        if (!buffer_.empty() || level_ == LogLevel::Error) {
            Logger::instance().log(level_, buffer_);
        }
    }

    LogMessage& LogMessage::operator<<(std::string_view value)
    {
        buffer_.append(value.data(), value.size());
        return *this;
    }

    LogMessage& LogMessage::operator<<(const char* value)
    {
        if (value) {
            buffer_ += value;
        }
        else {
            buffer_ += "(null)";
        }
        return *this;
    }

    LogMessage& LogMessage::operator<<(char value)
    {
        buffer_.push_back(value);
        return *this;
    }

    // =============================================================================
    // Logger implementation
    // =============================================================================

    Logger& Logger::instance()
    {
        static Logger instance;
        return instance;
    }

    void Logger::setLevel(LogLevel level)
    {
        std::lock_guard<std::mutex> lock(instance().mutex_);
        instance().minimumLevel_ = level;
    }

    void Logger::log(LogLevel level, const std::string& message)
    {
        Logger& logger = instance();

        // Check level threshold
        if (static_cast<int>(level) < static_cast<int>(logger.minimumLevel_)) {
            return;
        }

        std::lock_guard<std::mutex> lock(logger.mutex_);

        // Format: [LEVEL] [elapsed] message with ANSI colors
        const char* levelStr = "";
        const char* colorCode = "";

        switch (level) {
        case LogLevel::Debug:
            levelStr = "DEBUG";
            colorCode = "\033[36m";
            break; // Cyan
        case LogLevel::Info:
            levelStr = "INFO";
            colorCode = "\033[32m";
            break; // Green
        case LogLevel::Warning:
            levelStr = "WARN";
            colorCode = "\033[33m";
            break; // Yellow
        case LogLevel::Error:
            levelStr = "ERROR";
            colorCode = "\033[31m";
            break; // Red
        }

        std::cout << colorCode << "[" << levelStr << "]\033[0m"
                  << " [" << formatElapsed() << "] "
                  << message << std::endl;
    }

    std::chrono::milliseconds Logger::elapsedMillis()
    {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now - instance().startTime_);
    }

    std::string Logger::formatElapsed()
    {
        auto ms = elapsedMillis().count();

        std::ostringstream oss;
        if (ms < 1000) {
            oss << ms << "ms";
        }
        else if (ms < 60000) {
            oss << std::fixed << std::setprecision(2) << (ms / 1000.0) << "s";
        }
        else {
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
        , start_(std::chrono::steady_clock::now()) { }

    ScopedTimer::~ScopedTimer()
    {
        if (!stopped_) {
            stop();
        }
    }

    void ScopedTimer::stop()
    {
        if (!stopped_) {
            end_ = std::chrono::steady_clock::now();
            stopped_ = true;

            Logger::log(level_, label_ + " completed in " + getElapsedStr());
        }
    }

    double ScopedTimer::getElapsedSeconds() const
    {
        auto end = stopped_ ? end_ : std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start_);
        return duration.count();
    }

    std::string ScopedTimer::getElapsedStr() const
    {
        double seconds = getElapsedSeconds();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << seconds << "s";
        return oss.str();
    }

} // namespace mpfem
