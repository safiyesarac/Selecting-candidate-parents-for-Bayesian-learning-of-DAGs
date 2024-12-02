#pragma once

#include "common.h"
#include <type_traits>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <bitset>
#include <map>

template <typename T>
std::vector<T> read_symmetric_weights(const std::string& filename) {
    std::ifstream file;
    file.exceptions(file.failbit | file.badbit);
    file.open(filename);

    int size;
    file >> size;

    if (size <= 0) {
        std::cerr << "Invalid symmetric weight file\n";
        exit(1);
    }

    std::vector<T> weights(size);
    for (int i = 0; i < size; ++i) {
        double log_value;
        file >> log_value;
        if constexpr (std::is_floating_point<T>::value) {
            weights[i] = std::exp(log_value); // Directly using exp for double
        } else {
            weights[i] = T::from_log(log_value); // Assume T has from_log if not floating point
        }
    }

    return weights;
}
#include <unordered_map>
template <typename T>
using SparseWeights = std::unordered_map<size_t, T>;

template <typename T>
std::vector<SparseWeights<T>> read_nonsymmetric_weights(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        exit(1);
    }

    int size;
    file >> size;

    if (file.fail() || size <= 0) {
        std::cerr << "Invalid size in nonsymmetric weight file.\n" << size ;
        file.close();
        exit(1);
    }

    std::map<std::string, int> name_to_idx;
    for (int i = 0; i < size; ++i) {
        std::string name;
        int score_count;
        file >> name >> score_count;

        if (file.fail()) {
            std::cerr << "Error reading name or score_count at index " << i << ".\n";
            file.close();
            exit(1);
        }

        if (name_to_idx.count(name)) {
            std::cerr << "Duplicate name found in weight file: " << name << "\n";
            file.close();
            exit(1);
        }
        name_to_idx[name] = i;

        for (int j = 0; j < score_count; ++j) {
            double log_score;
            int parent_count;
            file >> log_score >> parent_count;

            for (int k = 0; k < parent_count; ++k) {
                std::string parent;
                file >> parent;
            }
        }
    }

    file.close();
    file.open(filename);

    int ignore;
    file >> ignore;

    std::vector<SparseWeights<T>> weights(size); // Sparse representation

    for (int i = 0; i < size; ++i) {
        std::string name;
        int score_count;
        file >> name >> score_count;

        for (int j = 0; j < score_count; ++j) {
            double log_score;
            int parent_count;
            file >> log_score >> parent_count;

            std::bitset<32> parents;
            for (int k = 0; k < parent_count; ++k) {
                std::string parent;
                file >> parent;
                auto it = name_to_idx.find(parent);
                if (it == name_to_idx.end() || it->second == i) {
                    std::cerr << "Invalid parent entry in weight file.\n";
                    file.close();
                    exit(1);
                }
                parents[it->second] = 1;
            }

            // Use parents.to_ulong() as the key for the sparse weights
            weights[i][parents.to_ulong()] = T(log_score);
        }
    }

    file.close();
    return weights;
}
