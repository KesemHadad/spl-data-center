#include "Request.h"
#include "SchedulingPolicy.h"
#include <algorithm>
#include <cstddef>

namespace {
struct ArrivalEntry {
    int admissionCycle;
    int index;
    int id;
};

bool arrivedFirst(const ArrivalEntry &a, const ArrivalEntry &b) {
    if (a.admissionCycle != b.admissionCycle)
        return a.admissionCycle < b.admissionCycle;
    return a.id < b.id;
}
}

std::vector<int>
ArrivalOrderPolicy::choose(const std::vector<const Request *> &eligible, const std::vector<GPU> &gpus) const {
    std::vector<ArrivalEntry> requestEntries(eligible.size());
    for (std::size_t i = 0; i < eligible.size(); i++)
        requestEntries[i] = ArrivalEntry{eligible[i]->getAdmissionCycle(), static_cast<int>(i), eligible[i]->getId()};

    std::sort(requestEntries.begin(), requestEntries.end(), arrivedFirst);

    std::vector<int> orderedRequestIndicies(eligible.size());
    for (std::size_t i = 0; i < requestEntries.size(); i++)
        orderedRequestIndicies[i] = requestEntries[i].index;

    return orderedRequestIndicies;
}

SchedulingPolicy *ArrivalOrderPolicy::clone() const {
    return new ArrivalOrderPolicy(*this);
}
