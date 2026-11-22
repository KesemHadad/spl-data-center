#include "GPU.h"
#include "Request.h"
#include "SchedulingPolicy.h"
#include <algorithm>
#include <cstddef>

namespace {
struct ShortestJobEntry {
    std::size_t blocks;
    int index;
    int id;
};

bool shorterJob(const ShortestJobEntry &a, const ShortestJobEntry &b) {
    if (a.blocks != b.blocks)
        return a.blocks < b.blocks;
    return a.id < b.id;
}
}

std::vector<int>
ShortestJobPolicy::choose(const std::vector<const Request *> &eligible, const std::vector<GPU> &gpus) const {
    std::vector<int> orderedRequestIndicies(eligible.size());

    if (gpus.empty()) {
        for (std::size_t i = 0; i < eligible.size(); i++)
            orderedRequestIndicies[i] = i;
        return orderedRequestIndicies;
    }

    std::vector<ShortestJobEntry> requestEntries(eligible.size());
    for (std::size_t i = 0; i < eligible.size(); i++)
        requestEntries[i] = ShortestJobEntry{gpus[0].getBlocksForMemory(eligible[i]->getMaxNeededMemory()), static_cast<int>(i), eligible[i]->getId()};

    std::sort(requestEntries.begin(), requestEntries.end(), shorterJob);

    for (std::size_t i = 0; i < requestEntries.size(); i++)
        orderedRequestIndicies[i] = requestEntries[i].index;

    return orderedRequestIndicies;
}

SchedulingPolicy *ShortestJobPolicy::clone() const {
    return new ShortestJobPolicy(*this);
}
