/**
 * @file exception.hpp
 * @brief Exception classes for mpfem
 */

#ifndef MPFEM_CORE_EXCEPTION_HPP
#define MPFEM_CORE_EXCEPTION_HPP

#include <sstream>
#include <stdexcept>
#include <string_view>

namespace mpfem {

/**
 * @brief Base exception class for mpfem
 */
class Exception : public std::runtime_error {
public:
    Exception(std::string_view file, int line, std::string_view message)
        : std::runtime_error("")
        , file_(file)
        , line_(line)
        , message_(message) {
    }

    const char* what() const noexcept override { return what_.c_str(); }

    const std::string& file() const { return file_; }
    int line() const { return line_; }
    const std::string& message() const { return message_; }

protected:
    std::string file_;
    int line_;
    std::string message_;
    std::string what_;
};

/**
 * @brief Exception for file I/O errors
 */
class FileError : public Exception {
public:
    FileError(std::string_view file, int line, std::string_view message)
        : Exception(file, line, message) {
        std::ostringstream oss;
        oss << "FileError at " << file_ << ':' << line_ << "\n  " << message;
        what_ = oss.str();
    }
};

/**
 * @brief Exception for invalid argument errors
 */
class InvalidArgument : public Exception {
public:
    InvalidArgument(std::string_view file, int line, std::string_view message)
        : Exception(file, line, message) {
        std::ostringstream oss;
        oss << "InvalidArgument at " << file_ << ':' << line_ << "\n  " << message;
        what_ = oss.str();
    }
};

/**
 * @brief Exception for runtime errors
 */
class RuntimeError : public Exception {
public:
    RuntimeError(std::string_view file, int line, std::string_view message)
        : Exception(file, line, message) {
        std::ostringstream oss;
        oss << "RuntimeError at " << file_ << ':' << line_ << "\n  " << message;
        what_ = oss.str();
    }
};

/**
 * @brief Exception for not implemented features
 */
class NotImplementedError : public Exception {
public:
    NotImplementedError(std::string_view file, int line, std::string_view message)
        : Exception(file, line, message) {
        std::ostringstream oss;
        oss << "NotImplementedError at " << file_ << ':' << line_ << "\n  " << message;
        what_ = oss.str();
    }
};

// ============================================================
// Exception throwing macros
// ============================================================

#define MPFEM_THROW(ExceptionType, msg) \
    do { \
        std::ostringstream _oss; \
        _oss << msg; \
        throw ExceptionType(__FILE__, __LINE__, _oss.str()); \
    } while(0)

}  // namespace mpfem

#endif  // MPFEM_CORE_EXCEPTION_HPP