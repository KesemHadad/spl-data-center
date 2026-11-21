#ifndef SCHEDULING_POLICY_H_
#define SCHEDULING_POLICY_H_
#include <vector>

class Request;
class GPU;

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    virtual std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const = 0;
    virtual SchedulingPolicy* clone() const = 0;
};

class ArrivalOrderPolicy : public SchedulingPolicy { 
    virtual std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const override;
    virtual SchedulingPolicy* clone() const override;

};
class ShortestJobPolicy : public SchedulingPolicy { 
    virtual std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const override;
    virtual SchedulingPolicy* clone() const override;
};
class LongestPrefixPolicy : public SchedulingPolicy {
    virtual std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const override;
    virtual SchedulingPolicy* clone() const override;
};


#endif