#include <iostream>
#include "json_pointer.hpp"

namespace demo {

const char* json_pointer::name() {
    return "json_pointer";
}

}  // namespace demo

int main() {
    std::cout << demo::json_pointer::name() << '\n';
    return 0;
}
