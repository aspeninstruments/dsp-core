#include "PerlinNoiseService.h"
#include <algorithm>
#include <cmath>

namespace dsp_core::Services {

double PerlinNoiseService::fade(double t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

double PerlinNoiseService::lerp(double a, double b, double t) {
    return a + t * (b - a);
}

double PerlinNoiseService::gradient1D(int x, unsigned int seed) {
    auto h = static_cast<unsigned int>(x);
    h ^= seed;
    h ^= (h >> 16);
    h *= 0x85ebca6b;
    h ^= (h >> 13);
    h *= 0xc2b2ae35;
    h ^= (h >> 16);
    return (static_cast<double>(h) / static_cast<double>(0xFFFFFFFF)) * 2.0 - 1.0;
}

double PerlinNoiseService::perlinNoise1D(double x, unsigned int seed) {
    int const x0 = static_cast<int>(std::floor(x));
    int const x1 = x0 + 1;
    double const localX = x - x0;
    double const g0 = gradient1D(x0, seed);
    double const g1 = gradient1D(x1, seed);
    double const d0 = g0 * localX;
    double const d1 = g1 * (localX - 1.0);
    double const t = fade(localX);
    return lerp(d0, d1, t);
}

double PerlinNoiseService::perlinFBM(double x, unsigned int seed, int octaves, double persistence) {
    double total = 0.0;
    double amplitude = 1.0;
    double maxAmplitude = 0.0;
    double frequency = 1.0;

    for (int i = 0; i < octaves; ++i) {
        total += perlinNoise1D(x * frequency, seed + static_cast<unsigned int>(i)) * amplitude;
        maxAmplitude += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }

    return total / maxAmplitude;
}

void PerlinNoiseService::alignZeroCrossing(std::vector<double>& noise, int tableSize) {
    const int numPoints = static_cast<int>(noise.size());

    if (numPoints != tableSize) {
        return;
    }

    const int originIndex = tableSize / 2;

    int bestZeroCrossingIdx = -1;
    int bestDistance = tableSize;

    for (int i = 0; i < numPoints - 1; ++i) {
        const double v0 = noise[static_cast<size_t>(i)];
        const double v1 = noise[static_cast<size_t>(i + 1)];

        if (v0 == 0.0 || v1 == 0.0 || (v0 * v1 < 0.0)) {
            const int dist = std::abs(i - originIndex);
            if (dist < bestDistance) {
                bestDistance = dist;
                bestZeroCrossingIdx = i;
            }
        }
    }

    if (bestZeroCrossingIdx >= 0) {
        const int shiftAmount = originIndex - bestZeroCrossingIdx;

        std::vector<double> shifted(static_cast<size_t>(numPoints));
        for (int i = 0; i < numPoints; ++i) {
            int sourceIdx = i - shiftAmount;
            while (sourceIdx < 0) {
                sourceIdx += numPoints;
            }
            while (sourceIdx >= numPoints) {
                sourceIdx -= numPoints;
            }
            shifted[static_cast<size_t>(i)] = noise[static_cast<size_t>(sourceIdx)];
        }

        noise = shifted;
    }

    noise[static_cast<size_t>(originIndex)] = 0.0;
}

} // namespace dsp_core::Services
