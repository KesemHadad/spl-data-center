#include "GPU.h"
#include "Model.h"
#include "Request.h"
#include "SchedulingPolicy.h"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace {
struct LongestPrefixEntry{
    std::size_t cachedPrefixLength;
    int index;
    int id;
};

std::size_t findLongestCachedPrefix(const Request &request, const std::vector<GPU> &gpus) {
    std::vector<std::string> cacheableTokens = request.getCacheableTokens();
    if (cacheableTokens.empty())
        return 0;

    std::size_t longestCachePrefixLength = 0;
    const Model &model = request.getModel();
    for (const GPU &gpu : gpus) {
        std::size_t matched = gpu.findLongestCachedPrefix(model.getModelId(), model.getEmbeddingSize(), cacheableTokens).matchedTokens;
        longestCachePrefixLength = std::max(longestCachePrefixLength, matched);
    }
    return longestCachePrefixLength;
}

bool longerPrefixEntry(const LongestPrefixEntry &a, const LongestPrefixEntry &b) {
    if (a.cachedPrefixLength != b.cachedPrefixLength)
        return a.cachedPrefixLength > b.cachedPrefixLength;
    return a.id < b.id;
}
}

std::vector<int>
LongestPrefixPolicy::choose(const std::vector<const Request *> &eligible, const std::vector<GPU> &gpus) const {
    std::vector<int> orderedRequestIndicies(eligible.size());

    std::vector<LongestPrefixEntry> requestEntries(eligible.size());
    for (std::size_t i = 0; i < eligible.size(); i++)
        requestEntries[i] =
            LongestPrefixEntry{findLongestCachedPrefix(*eligible[i], gpus), static_cast<int>(i), eligible[i]->getId()};

    std::sort(requestEntries.begin(), requestEntries.end(), longerPrefixEntry);

    for (std::size_t i = 0; i < requestEntries.size(); i++)
        orderedRequestIndicies[i] = requestEntries[i].index;

    return orderedRequestIndicies;
}

SchedulingPolicy *LongestPrefixPolicy::clone() const{
    return new LongestPrefixPolicy(*this);
}
