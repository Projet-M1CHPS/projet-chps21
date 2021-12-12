#include <version.hpp>

namespace tscl {

  Version Version::_current = Version(0, 0, 0, "undefined");
  Version const &Version::current = Version::_current;

  bool Version::operator>(const Version &other) const {
    return (major > other.major) or (major == other.major and minor > other.minor) or
           (major == other.major and minor == other.minor and patch > other.patch and
            (tweak.empty() or tweak > other.tweak));
  }

  bool Version::operator<(const tscl::Version &other) const {
    return not(*this == other) and not(*this > other);
  }

}   // namespace tscl