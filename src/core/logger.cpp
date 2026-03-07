#include "mpfem/core/logger.hpp"

#include <cstdarg>
#include <ctime>

#ifdef _WIN32
#define localtime_r(t, tm) localtime_s(tm, t)
#endif

namespace mpfem {

Logger& Logger::Instance() {
  static Logger instance;
  return instance;
}

Logger::~Logger() { CloseFile(); }

void Logger::SetFile(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (file_) {
    std::fclose(file_);
  }
  file_ = std::fopen(path.c_str(), "a");
}

void Logger::CloseFile() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (file_) {
    std::fclose(file_);
    file_ = nullptr;
  }
}

void Logger::Log(LogLevel level, const char* file, int line, const char* func,
                 const char* format, ...) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::FILE* out = file_ ? file_ : stdout;

  FormatTime(out);
  std::fprintf(out, " [%s] %s:%d %s: ", LevelName(level), file, line, func);

  va_list args;
  va_start(args, format);
  std::vfprintf(out, format, args);
  va_end(args);

  std::fprintf(out, "\n");
  std::fflush(out);
}

void Logger::FormatTime(std::FILE* out) {
  const auto now = std::time(nullptr);
  std::tm tm_buf;
  localtime_r(&now, &tm_buf);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
  std::fprintf(out, "%s", buf);
}

const char* Logger::LevelName(LogLevel level) const {
  switch (level) {
    case LogLevel::kTrace:
      return "TRACE";
    case LogLevel::kDebug:
      return "DEBUG";
    case LogLevel::kInfo:
      return "INFO";
    case LogLevel::kWarn:
      return "WARN";
    case LogLevel::kError:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
}

}  // namespace mpfem
