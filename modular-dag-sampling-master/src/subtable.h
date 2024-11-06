#ifndef SUBTABLE_H
#define SUBTABLE_H

#include <vector>

template <typename T>
class SubTable {
public:
    SubTable() = default;  // Nullary constructor
    SubTable(uint32_t n) : data(1 << n, std::vector<T>(1 << n, T(0))) {}

    T& operator()(uint32_t R, uint32_t U) { return data[R][U]; }
    const T& operator()(uint32_t R, uint32_t U) const { return data[R][U]; }

    // Add a public getter for `data`
    const std::vector<std::vector<T>>& get_data() const { return data; }

private:
    std::vector<std::vector<T>> data;
};

#endif  // SUBTABLE_H
