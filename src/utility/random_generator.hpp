#ifndef RANDOM_GENERATOR
#define RANDOM_GENERATOR

#include <random>

class RandomGenerator
{
public:
    static RandomGenerator& getInstance()
    {
        static RandomGenerator instance;
        return instance;
    };

    double random_real(double min = 0.0, double max = 1.0)
    {
        return real_dist(gen) * (max - min) + min;
    }

    void set_seed(int seed)
    {
        gen = std::mt19937(seed);
    }
private:
    RandomGenerator()
    {
        gen = std::mt19937(device());
    }

    std::random_device device;
    std::mt19937 gen;
    std::uniform_real_distribution <double> real_dist;
};

#endif
