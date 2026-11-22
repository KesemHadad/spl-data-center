#ifndef GPU_H_
#define GPU_H_
#include <map>

#include <string>
#include <cstddef>
#include <vector>
#include "Request.h"

struct PrefixMatch {
    std::size_t matchedTokens;
    std::vector<std::size_t> blockIds;
};

class MemoryBlock;
class Request; 
class GenerativeRequest;
class EmbeddingRequest;

class GPU{
public:
    GPU(std::size_t blocksNum, std::size_t blockSize);
    std::size_t availableSlots() const;
    std::size_t getBlocksForMemory(std::size_t memory) const;
    void reserve(std::size_t n, const Request& request);
    void unreserve(std::size_t n, const Request& request);
    std::size_t allocateBlock(const Request& request);
    void addMemoryBlock(const GenerativeRequest& request, std::size_t slot);
    void addMemoryBlock(const EmbeddingRequest& request, std::size_t slot);
    void acquire(std::size_t slot);
    void release(std::size_t slot);
    void copyMemoryFromOtherGPU(const GPU& other,const std::vector<std::size_t>& blocks);
    void assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId);
    int readFromBlock(std::size_t blockId, std::size_t offset) const;
    void writeToBlock(std::size_t blockId, std::size_t offset, int value);
    PrefixMatch findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const;

    //For Server and Request to access private members of GPU - Kesem added:
    std::size_t getBlockSize() const;        // Returns blockSize (number of integers per block).
    bool isSlotIdle(std::size_t slot) const; // Returns true if the slot holds a non-null block with refCount == 0 (idle).
    std::size_t getCacheHits() const;        // Returns total cached blocks acquired at admission (including copied blocks).
    std::size_t getEvictions() const;        // Returns total idle blocks evicted to make room for new or copied blocks.
private:
    std::size_t blocksNum;
    std::size_t blockSize;
    std::vector<MemoryBlock*> memory;

    std::size_t reservedTotal;
    std::map<int, std::size_t> reservations;
};

#endif