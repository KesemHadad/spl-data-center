#include "GPU.h"
#include "MemoryBlock.h"



// Constructor
GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr),
      reservedTotal(0), reservations(), cacheMetadata(blocksNum), 
      cacheHitCount(0), evictCount(0) {

}

// Get Block Size
std::size_t GPU::getBlockSize() const { return this->blockSize; }

// Get Blocks For Memory
std::size_t GPU::getBlocksForMemory(std::size_t memory) const {
    if (memory % blockSize != 0)
        return (memory/blockSize) + 1;
    return memory/blockSize;
}

bool GPU::isSlotIdle(std::size_t slot) const {
    return (memory[slot] != nullptr && memory[slot]->getRefCount() == 0);
}

// Assumes BlockId, memory[BlockID] are valid and offset < blockSize
int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value ) {
    memory[blockId]->writeToBlock(offset, value);
}



std::size_t GPU::availableSlots() const {
    size_t active_blocks = 0;

    for (size_t i =0; i < blocksNum; i=i+1) {
        if (memory[i] != nullptr && memory[i]->getRefCount() > 0) {
            active_blocks = active_blocks + 1;
        }
    }
    return (blocksNum - active_blocks - reservedTotal);
}

// עבור בקשה חדשה מאתחל לפי id ב-0 ומוסיף n אוטומטית
void GPU::reserve(std::size_t n, const Request& request) {
    reservations[request.getId()] += n;
    reservedTotal += n;
}

void GPU::unreserve(std::size_t n, const Request& request) {
    int id = request.getId();
    reservations[id] = reservations[id] - n;
    reservedTotal = reservedTotal - n;
    if (reservations[id] == 0)
        reservations.erase(id);
}



// checks for an existing entry with the same modelId and prefixTokens, assigns prefix via cache struct.
void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId) {
    bool alreadyExist = false;
    for (const CacheEntry& entry : cacheMetadata) {
        if (entry.hasPrefix &&
            entry.modelId == modelId &&
            entry.prefixTokens == prefixTokens) {
            alreadyExist = true;
            break;
        }
    }

    if (!alreadyExist) {
        cacheMetadata[block].modelId = modelId;
        cacheMetadata[block].prefixTokens = prefixTokens;
        cacheMetadata[block].hasPrefix = true;
    }
}

void GPU::release(std::size_t slot) {
    IdleAction action = memory[slot]->release();
    if (memory[slot]->getRefCount() == 0) {
        if ((action == IdleAction::KeepCached && !cacheMetadata[slot].hasPrefix) ||
            action == IdleAction::Destroy) {
                delete memory[slot];
                memory[slot] = nullptr;
                cacheMetadata[slot] = CacheEntry();
            }
    }
}

void GPU::acquire(std::size_t slot) {
    memory[slot]->acquire();
    if (cacheMetadata[slot].hasPrefix) {
        cacheHitCount = cacheHitCount + 1;
    }
}


