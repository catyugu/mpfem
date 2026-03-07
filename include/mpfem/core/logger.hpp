#pragma once

#include <cstdio>
#include <mutex>
#include <string>

namespace mpfem {

enum class LogLevel {
  kTrace = 0,
  kDebug = 1,
  kInfo = 2,
  kWarn = 3,
  kError = 4,
  kOff = 5
};

class Logger {
 public:
  static Logger& Instance();

  void SetLevel(LogLevel level) { level_ = level; }
  LogLevel GetLevel() const { return level_; }

  void SetFile(const std::string& path);
  void CloseFile();

  void Log(LogLevel level, const char* file, int line, const char* func,
            const char* format, ...);

  bool IsEnabled(LogLevel level) const { return level >= level_; }

 private:
  Logger() = default;
  ~Logger();

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  void FormatTime(std::FILE* out);
  const char* LevelName(LogLevel level) const;

  LogLevel level_ = LogLevel::kInfo;
  std::FILE* file_ = nullptr;
  std::mutex mutex_;
};

}  // namespace mpfem

#define MPFEM_LOG_IMPL(level, ...)                                            \
  do {                                                                        \
    if (::mpfem::Logger::Instance().IsEnabled(level)) {                       \
      ::mpfem::Logger::Instance().Log(level, __FILE__, __LINE__, __func__,    \
                                      __VA_ARGS__);                           \
    }                                                                         \
  } while (0)

#define MPFEM_TRACE(...) MPFEM_LOG_IMPL(::mpfem::LogLevel::kTrace, __VA_ARGS__)
#define MPFEM_DEBUG(...) MPFEM_LOG_IMPL(::mpfem::LogLevel::kDebug, __VA_ARGS__)
#define MPFEM_INFO(...) MPFEM_LOG_IMPL(::mpfem::LogLevel::kInfo, __VA_ARGS__)
#define MPFEM_WARN(...) MPFEM_LOG_IMPL(::mpfem::LogLevel::kWarn, __VA_ARGS__)
#define MPFEM_ERROR(...) MPFEM_LOG_IMPL(::mpfem::LogLevel::kError, __VA_ARGS__)
