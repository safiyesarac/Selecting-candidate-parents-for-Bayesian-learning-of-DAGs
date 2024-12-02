#ifndef NONSYMMETRIC_H
#define NONSYMMETRIC_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <map>
#include <cmath>  // for std::log
#include "common.h"
#include "lognum.h"
#include "subtable.h"
#include "symmetric.h"
#include "readwrite.h"


namespace nonsymmetric_ {
    double uniform_rand(double upper_bound) {
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<> dist(0.0, upper_bound);
    return dist(gen);
}

void write_dags(const std::vector<std::vector<int>>& dags) {
    for(const std::vector<int>& dag : dags) {
        int size = dag.size();

        for(int i = 0; i < size; ++i) {
            if(i) {
                std::cout << ", ";
            }

            std::cout << i << " <- {";

            bool first = true;
            for(int j = 0; j < size; ++j) {
                if(dag[i] & (1 << j)) {
                    if(!first) {
                        std::cout << ", ";
                    }
                    first = false;
                    std::cout << j;
                }
            }
            std::cout << "}";
        }
        std::cout << "\n";
    }
}


double powi(double base, int exponent) {
    return std::pow(base, exponent);
}

double binomial(int n, int k) {
    double res = 1.0;
    for (int i = 1; i <= k; ++i) {
        res *= (n - i + 1) / static_cast<double>(i);
    }
    return res;
}

template <typename T>
std::vector<SubTable<T>> calculate_hat_weights(int size, const std::vector<std::vector<T>>& weights) {
     std::vector<std::vector<T>> hat_weights_1;
    std::vector<SubTable<T>> hat_weights_2;

    for (int i = 0; i < size+1; ++i)
    {
        std::vector<T> vectA1(((size_t)1 << size), T(0));
        SubTable<T> vectB1(size);
        hat_weights_1.push_back(vectA1);
        hat_weights_2.push_back(vectB1);
    }


    for (int i = 0; i < size; ++i)
    {
        std::bitset<32> V(((size_t)1 << size)-1);
        V[i] = 0;
        int V_sub_i = (int)V.to_ulong();

        hat_weights_1[i][0] = weights[i][0];
        hat_weights_2[i](0, 0) = weights[i][0];

        //Go through all nonempty subsets of V\{i};
        for (int t = 0; (t=(t-V_sub_i)&V_sub_i);)
        {
            T sum_1 = weights[i][0];
            for (int S1 = 0; (S1=(S1-t)&t);)
            {
                sum_1 = sum_1 + weights[i][S1];
            }
            hat_weights_1[i][t] = sum_1;

            /*SINGLETON CASE*/
            for(int p = 0; p < size; p++) {
                int node = (size_t)1 << p;
                if((t&node) != 0) {
                    T sum_2 = T(0);
                    for(int S = 0; (S=(S-t)&t);) {
                        if((S&node) != 0) {
                            sum_2 = sum_2 + weights[i][S];
                        }
                    }
                    hat_weights_2[i](node, t) = sum_2;
                }
            }

            for(int R = 0; (R=(R-t)&t);) {
                T sum = T(0);
                std::bitset<32> R_bits(R);
                int previous = 0;
                for(int k = 0; k < size; k++) {
                    if(R_bits[k] == 1) {
                        int k_num = (size_t)1 << k;
                        int set = t;

                        set = set&(~previous);
                        previous = previous|k_num;
                        sum = sum + hat_weights_2[i](k_num, set);
                    }
                }
                hat_weights_2[i](R, t) = sum;
            }

            hat_weights_2[i](0,t) = hat_weights_1[i][t];
        }
    }

    return hat_weights_2;
}

template <class T>
SubTable<T> monotone_calculate_fs(int size, const std::vector<SubTable<T>>& hws) {
   SubTable<T> fs(size);

    fs(0, 0) = T(0);

    std::bitset<32> V(((size_t)1 << size)-1);
    int V_sub = (int) V.to_ulong();

    for (int U=0; (U=(U-V_sub)&V_sub);) {
        for (int S_0 = 0; (S_0=(S_0-U)&U);) {
        int upmask = U&(~S_0);
        if(S_0 == U) {
            fs(S_0,U) = T(1);
        } else {
            T sum1 = T(0);
            for (int S_1 = 0; (S_1=(S_1-upmask)&upmask);) {

                T product = T(1);
                std::bitset<32> S_1_bits(S_1);

                for (int i = 0; i < size; i++) {
                    if(S_1_bits[i] == 1) {
                        product = product*hws[i](S_0, V_sub&(~upmask));
                    }
                }
                product = product*fs(S_1, U&(~S_0));
                sum1 = sum1 + product;
            }
            fs(S_0, U) = sum1;
            }
        }
    }

    return fs;
}

template<typename T>
class NonSymmetricSampler {
public:
    using WeightT = std::vector<std::vector<T>>;
    const std::string filename;

    NonSymmetricSampler(const std::string& filename) : filename(filename) {
        preprocess();
    }

    std::vector<int> sample() const {
        using namespace nonsymmetric_;
    
    // Log the start of the sampling process
    std::cout << "Starting sample method." << std::endl;
    
    // Step 1: Sample layering
    std::vector<int> layering = sample_layering(weights.size(), hat_weights, rus);
    
    // Log the resulting layering
    std::cout << "Layering sampled:" << std::endl;
    for (size_t i = 0; i < layering.size(); ++i) {
        std::cout << "layering[" << i << "] = " << layering[i] << std::endl;
    }

    // Step 2: Sample parents based on the layering
    std::vector<int> dag = sample_parents_ns(weights.size(), layering, weights);

    // Log the resulting DAG structure
    std::cout << "DAG structure sampled:" << std::endl;
    for (size_t i = 0; i < dag.size(); ++i) {
        std::cout << "dag[" << i << "] = " << std::bitset<32>(dag[i]) << std::endl;
    }

    // Log completion of the sampling process
    std::cout << "Sample method completed." << std::endl;
    std::vector<std::vector<int>> dags;
    for (int i = 0; i < 1; ++i) {
        dags.push_back(dag);
    }

    write_dags(dags);
    return dag;
    }
std::vector<int> sample_layering(int size, const std::vector<SubTable<T>>& hws, SubTable<T> fs) const {
	/*
	Section 3.2. */
	std::vector<int> layering;
    layering.push_back(0);

    int partition_count = 0;
    int j = 1;
    int previous_rs = 0;
    int U;
    int V = ((size_t)1 << size) - 1;

    std::cout << "Starting sample_layering method." << std::endl;

    while (partition_count < size) {
        U = V & (~previous_rs);
        std::bitset<32> U_bits(U);
        T upper_bound = T(0);

        std::vector<T> bounds(U + 1, T(0));
        std::cout << "Iteration " << j << ": U = " << U_bits << ", previous_rs = " << std::bitset<32>(previous_rs) << std::endl;

        for (int R = 0; (R = (R - U) & U);) {
            T product = T(0);
            T inner_product = T(1);
            int r_k_union = 0;

            // Calculate inner_product for each layer
            for (int k = 1; k <= j - 1; k++) {
                int r_mk = layering[k - 1];
                r_k_union = r_k_union | r_mk;

                std::bitset<32> r_k_bits(layering[k]);
                for (int i = 0; i < size; i++) {
                    if (r_k_bits[i] == 1) {
                        T prev = hws[i](r_mk, r_k_union);
                        inner_product = inner_product * prev;
                    }
                }
            }

            product = inner_product;
            int r_mj = layering[j - 1];
            std::bitset<32> r_bits(R);

            // Calculate product for each element in R
            for (int i = 0; i < size; i++) {
                if (r_bits[i] == 1) {
                    T prev = hws[i](r_mj, V & (~U));
                    product = product * prev;
                }
            }

            product = product * fs(R, U);
            upper_bound = upper_bound + product;
            bounds[R] = upper_bound;

            std::cout << "Bounds for R=" << r_bits << ": product=" << product << ", upper_bound=" << upper_bound << std::endl;
        }

        T random_number = uniform_rand(upper_bound);
        std::cout << "Random number for selection: " << random_number << std::endl;

        for (int R = 0; (R = (R - U) & U);) {
            if (bounds[R] > random_number) {
                layering.push_back(R);
                previous_rs = previous_rs | R;
                std::cout << "Selected R=" << std::bitset<32>(R) << " for layering." << std::endl;
                break;
            }
        }

        std::bitset<32> added(layering[j]);
        for (int i = 0; i < size; i++) {
            if (added[i] == 1) {
                partition_count++;
            }
        }
        std::cout << "Layering after iteration " << j << ": " << layering.back() << ", partition_count=" << partition_count << std::endl;

        j++;
    }

    std::cout << "Final layering: ";
    for (const auto& layer : layering) {
        std::cout << layer << " ";
    }
    std::cout << std::endl;

    return layering;
}

std::vector<int> sample_parents_ns(int size, const std::vector<int>& layering,
    const std::vector<std::vector<T>>& weights) const {
	/*
	Section 3.2.
	*/


    std::vector<int> dag(size, 0);

    int U = 0;
    U = U|layering[1];

    int previous_partition = U;

    for(int j = 2; j < (int) layering.size(); j ++) {
        std::bitset<32> layer_bits(layering[j]);
        for(int p = 0; p < size; p++) {
            if(layer_bits[p] == 0) {
                continue;
            }
            int node = p;
            T upper_bound = T(0);
            for(int G = 0; (G=(G-U)&U);) {
                if((G&previous_partition) != 0) {
                    upper_bound = upper_bound + weights[node][G];
                }
            }

            T random = uniform_rand(upper_bound);
            T cumulative = T(0);
            for(int G = 0; (G=(G-U)&U);) {
                if((G&previous_partition) != 0) {
                    cumulative = cumulative + weights[node][G];
                    if(cumulative > random) {
                        std::bitset<32> parent_bits(G);
                        for (int i = 0; i < size; i++) {
                            if(parent_bits[i] == 1) {
                                dag[node] |= 1 << i;
                            }
                        }
                        break;
                    }
                }
            }
        }
        previous_partition = layering[j];
        U = U|previous_partition;
    }

    return dag;


}


private:
    

    WeightT weights;
    std::vector<SubTable<T>> hat_weights;
    SubTable<T> rus;

    std::vector<std::vector<T>> convertToDense(const std::vector<std::unordered_map<size_t, T>>& sparseWeights, int size) {
        std::vector<std::vector<T>> denseWeights(size, std::vector<T>(1 << size, T(0)));

        for (size_t i = 0; i < sparseWeights.size(); ++i) {
            for (const auto& kv : sparseWeights[i]) {  // Avoid structured bindings for C++14 compatibility
                size_t key = kv.first;
                T value = kv.second;
                denseWeights[i][key] = value;
            }
        }

        return denseWeights;
    }


    void preprocess() {
        std::cout << "Using filename: " << filename << std::endl;
        auto sparseWeights = read_nonsymmetric_weights<T>(filename);
        int size = sparseWeights.size();  // Define size from sparseWeights
        weights = convertToDense(sparseWeights, size);

        hat_weights = calculate_hat_weights<T>(weights.size(), weights);

    // Debug: log contents of hat_weights
    std::cout << "Debug: hat_weights calculated:" << std::endl;
    for (int i = 0; i < hat_weights.size(); ++i) {
        std::cout << "hat_weights[" << i << "]: ";
        // Assuming SubTable has a way to iterate or access its contents
        for (int j = 0; j < (1 << size); ++j) {
            for (int k = 0; k < (1 << size); ++k) {
                std::cout << hat_weights[i](j, k) << " ";
            }
        }
        std::cout << std::endl;
    }

    // Calculate rus
    rus = monotone_calculate_fs<T>(weights.size(), hat_weights);

    // Debug: log final rus values
    std::cout << "Debug: rus calculated:" << std::endl;
    for (int i = 0; i < (1 << size); ++i) {
        for (int j = 0; j < (1 << size); ++j) {
            std::cout << rus(i, j) << " ";
        }
        std::cout << std::endl;
    }
    }

    static std::vector<T> flatten(const WeightT& matrix) {
        std::vector<T> flat;
        for (const auto& row : matrix) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
        return flat;
    }

    static std::vector<std::vector<T>> convertToVector(const std::vector<SubTable<T>>& tables) {
        std::vector<std::vector<T>> converted;
        for (const auto& table : tables) {
            std::vector<T> row;
            // Add alternative access logic here based on `SubTable`'s access methods
            converted.push_back(row);
        }
        return converted;
    }
};

}  // namespace nonsymmetric_

#endif  // NONSYMMETRIC_H
