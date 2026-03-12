#ifndef MPFEM_EXCEPTION_HPP
#define MPFEM_EXCEPTION_HPP

#include <stdexcept>
#include <string>
#include <source_location>

namespace mpfem {

/// Base exception class for mpfem
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& message,
                       const std::source_location& loc = std::source_location::current())
        : std::runtime_error(formatMessage(message, loc)) {}

private:
    static std::string formatMessage(const std::string& msg, 
                                      const std::source_location& loc) {
        return "[" + std::string(loc.file_name()) + ":" + 
               std::to_string(loc.line()) + "] " + msg;
    }
};

/// Exception for file I/O errors
class FileException : public Exception {
public:
    explicit FileException(const std::string& message)
        : Exception("File error: " + message) {}
};

/// Exception for mesh parsing errors
class MeshException : public Exception {
public:
    explicit MeshException(const std::string& message)
        : Exception("Mesh error: " + message) {}
};

/// Exception for finite element errors
class FeException : public Exception {
public:
    explicit FeException(const std::string& message)
        : Exception("Finite element error: " + message) {}
};

/// Exception for solver errors
class SolverException : public Exception {
public:
    explicit SolverException(const std::string& message)
        : Exception("Solver error: " + message) {}
};

/// Exception for invalid arguments
class ArgumentException : public Exception {
public:
    explicit ArgumentException(const std::string& message)
        : Exception("Invalid argument: " + message) {}
};

/// Exception for out-of-range access
class RangeException : public Exception {
public:
    explicit RangeException(const std::string& message)
        : Exception("Range error: " + message) {}
};

/// Exception for not-yet-implemented features
class NotImplementedException : public Exception {
public:
    explicit NotImplementedException(const std::string& feature)
        : Exception("Not implemented: " + feature) {}
};

}  // namespace mpfem

// =============================================================================
// Convenience macros
// =============================================================================

#define MPFEM_THROW(type, msg) throw ::mpfem::type(msg)

#define MPFEM_ASSERT(cond, msg)                    \
    do {                                           \
        if (!(cond)) {                             \
            MPFEM_THROW(Exception, msg);           \
        }                                          \
    } while (0)

#endif  // MPFEM_EXCEPTION_HPP
