/**
 * @file exception.hpp
 * @brief Exception types for mpfem
 */

#ifndef MPFEM_CORE_EXCEPTION_HPP
#define MPFEM_CORE_EXCEPTION_HPP

#include <exception>
#include <sstream>
#include <string>
#include <string_view>

namespace mpfem {

// ============================================================
// Base Exception
// ============================================================

/**
 * @brief Base exception class for mpfem
 */
class Exception : public std::exception {
public:
    Exception(std::string_view file, int line, std::string_view message)
        : file_(file), line_(line) {
        std::ostringstream oss;
        oss << "mpfem::Exception at " << file_ << ':' << line_ << "\n  "
            << message;
        what_ = oss.str();
    }

    const char* what() const noexcept override { return what_.c_str(); }

    const std::string& file() const { return file_; }
    int line() const { return line_; }

protected:
    std::string file_;
    int line_;
    std::string what_;
};

// ============================================================
// Specific Exception Types
// ============================================================

/**
 * @brief Exception for invalid argument values
 */
class InvalidArgument : public Exception {
public:
    InvalidArgument(std::string_view file, int line, std::string_view message)
        : Exception(file, line, message) {
        std::ostringstream oss;
        oss << "InvalidArgument at " << file_ << ':' << line_ << "\n  "
            << message;
        what_ = oss.str();
    }
};

/**
 * @brief Exception for out-of-range access
 */
class OutOfRange : public Exception {
public:
    OutOfRange(std::string_view file, int line, std::string_view message)
        : Exception(file, line, message) {
        std::ostringstream oss;
        oss << "OutOfRange at " << file_ << ':' << line_ << "\n  " << message;
        what_ = oss.str();
    }
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
 * @brief Exception for mesh-related errors
 */
class MeshError : public Exception {
public:
    MeshError(std::string_view file, int line, std::string_view message)
        : Exception(file, line, message) {
        std::ostringstream oss;
        oss << "MeshError at " << file_ << ':' << line_ << "\n  " << message;
        what_ = oss.str();
    }
};

/**
 * @brief Exception for finite element errors
 */
class FEError : public Exception {
public:
    FEError(std::string_view file, int line, std::string_view message)
        : Exception(file, line, message) {
        std::ostringstream oss;
        oss << "FEError at " << file_ << ':' << line_ << "\n  " << message;
        what_ = oss.str();
    }
};

/**
 * @brief Exception for solver errors
 */
class SolverError : public Exception {
public:
    SolverError(std::string_view file, int line, std::string_view message)
        : Exception(file, line, message) {
        std::ostringstream oss;
        oss << "SolverError at " << file_ << ':' << line_ << "\n  " << message;
        what_ = oss.str();
    }
};

/**
 * @brief Exception for configuration errors
 */
class ConfigError : public Exception {
public:
    ConfigError(std::string_view file, int line, std::string_view message)
        : Exception(file, line, message) {
        std::ostringstream oss;
        oss << "ConfigError at " << file_ << ':' << line_ << "\n  " << message;
        what_ = oss.str();
    }
};

// ============================================================
// Throw Macros
// ============================================================

#define MPFEM_THROW(exception_type, msg)                           \
    do {                                                           \
        std::ostringstream _oss;                                   \
        _oss << msg;                                               \
        throw exception_type(__FILE__, __LINE__, _oss.str());      \
    } while (0)

#define MPFEM_THROW_IF(cond, exception_type, msg)                  \
    do {                                                           \
        if (cond) {                                                \
            MPFEM_THROW(exception_type, msg);                      \
        }                                                          \
    } while (0)

// Convenience macros for common exceptions
#define MPFEM_ASSERT(cond, msg)                                    \
    MPFEM_THROW_IF(!(cond), mpfem::InvalidArgument, msg)

#define MPFEM_ASSERT_RANGE(idx, min, max)                          \
    MPFEM_THROW_IF((idx) < (min) || (idx) >= (max),                \
                   mpfem::OutOfRange,                              \
                   "Index " << (idx) << " out of range ["          \
                             << (min) << ", " << (max) << ")")

}  // namespace mpfem

#endif  // MPFEM_CORE_EXCEPTION_HPP
