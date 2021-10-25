#pragma once

namespace utils {

    void error(const std::string& msg) noexcept;

    void error(const char* msg) noexcept;

    // Generic IO exception
    class IOException : public std::runtime_error {
        public:
            IOException(const std::string& msg) noexcept;
            IOException(const char* msg) noexcept;
    };

}