#pragma once

#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>

namespace mpfem {

/// 日志级别枚举
enum class LogLevel {
  kTrace = 0,  ///< 详细跟踪信息，用于深度调试
  kDebug = 1,  ///< 调试信息
  kInfo = 2,   ///< 一般信息
  kWarn = 3,   ///< 警告信息
  kError = 4,  ///< 错误信息
  kOff = 5     ///< 关闭日志
};

/// 线程安全的日志系统
///
/// 支持多级别日志输出、文件输出、控制台输出。
/// 使用单例模式，通过宏简化调用。
///
/// 示例:
/// @code
/// MPFEM_INFO("Starting computation with %d elements", n_elements);
/// MPFEM_ERROR("Failed to open file: %s", filename);
/// @endcode
class Logger {
 public:
  /// 获取单例实例
  static Logger& Instance();

  /// 设置日志级别
  void SetLevel(LogLevel level) { level_ = level; }

  /// 获取当前日志级别
  LogLevel GetLevel() const { return level_; }

  /// 设置日志输出文件
  /// @param path 日志文件路径
  void SetFile(const std::string& path);

  /// 关闭日志文件
  void CloseFile();

  /// 输出日志
  void Log(LogLevel level, const char* file, int line, const char* func,
            const char* format, ...);

  /// 检查指定级别是否启用
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

/// 作用域计时器，在析构时自动输出耗时
///
/// 配合 MPFEM_TIMER 宏使用，用于性能分析。
class ScopedTimer {
 public:
  ScopedTimer(const char* name, const char* file, int line)
      : name_(name), file_(file), line_(line) {
    start_ = std::chrono::high_resolution_clock::now();
  }

  ~ScopedTimer() {
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
    Logger::Instance().Log(LogLevel::kInfo, file_, line_, "",
                           "[TIMER] %s: %lld ms", name_,
                           static_cast<long long>(duration.count()));
  }

 private:
  const char* name_;
  const char* file_;
  int line_;
  std::chrono::high_resolution_clock::time_point start_;
};

}  // namespace mpfem

#define MPFEM_LOG_IMPL(level, ...)                                            \
  do {                                                                        \
    if (::mpfem::Logger::Instance().IsEnabled(level)) {                       \
      ::mpfem::Logger::Instance().Log(level, __FILE__, __LINE__, __func__,    \
                                      __VA_ARGS__);                           \
    }                                                                         \
  } while (0)

/// 输出跟踪级别日志
#define MPFEM_TRACE(...) MPFEM_LOG_IMPL(::mpfem::LogLevel::kTrace, __VA_ARGS__)
/// 输出调试级别日志
#define MPFEM_DEBUG(...) MPFEM_LOG_IMPL(::mpfem::LogLevel::kDebug, __VA_ARGS__)
/// 输出信息级别日志
#define MPFEM_INFO(...) MPFEM_LOG_IMPL(::mpfem::LogLevel::kInfo, __VA_ARGS__)
/// 输出警告级别日志
#define MPFEM_WARN(...) MPFEM_LOG_IMPL(::mpfem::LogLevel::kWarn, __VA_ARGS__)
/// 输出错误级别日志
#define MPFEM_ERROR(...) MPFEM_LOG_IMPL(::mpfem::LogLevel::kError, __VA_ARGS__)

/// 作用域计时器宏，自动输出代码块耗时
/// @param name 计时器名称，用于标识
///
/// 示例:
/// @code
/// void Compute() {
///   MPFEM_TIMER("MatrixAssembly");
///   // ... 矩阵组装代码
/// }  // 作用域结束，自动输出耗时
/// @endcode
#define MPFEM_TIMER(name)                                                     \
  ::mpfem::ScopedTimer _mpfem_timer_##name(#name, __FILE__, __LINE__)
