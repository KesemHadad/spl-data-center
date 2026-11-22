#include "GPU.h"
#include "MemoryBlock.h"
#include <cassert>



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

//============================================================================================================

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

//============================================================================================================

// checks for an existing entry with the same modelId and prefixTokens, assigns prefix via cache struct.
void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId) {
    bool alreadyExist = false;
    for (const CacheEntry& entry : cacheMetadata) {
        if (entry.hasPrefix && entry.modelId == modelId &&
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

//============================================================================================================

 // GenerativeRequest
void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot) {
    memory[slot] = new KVBlock(blockSize);
}

// EmbeddingRequest
void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot) {
    memory[slot] = new EmbeddingWorkspace(blockSize);
}

std::size_t GPU::allocateBlock(const Request& request) {
    assert(reservations.count(request.getId()) > 0 &&
       reservations.at(request.getId()) > 0);

    size_t slot = blocksNum; // ערך התחלתי משמעותו - לא נמצא סלוט מתאים עדיין

    //find first empty
    for (size_t i = 0; i < blocksNum && slot == blocksNum; i=i+1) {
        if (memory[i] == nullptr) {
            slot = i;
        }
    }

    // if none empty find first idle
    if (slot == blocksNum) {
        for (size_t i = 0; i < blocksNum && slot == blocksNum; i=i+1) {
            if (isSlotIdle) {
                slot = i;
            }
        }
    }

    assert(slot < blocksNum);
    if (memory[slot] != nullptr) {
        evictCount = evictCount + 1;
        delete memory[slot];
        cacheMetadata[slot] = CacheEntry();
    }

    request.allocateOneBlock(*this, slot); // לא הבנתי למה אני קוראת לרקווסט ולא עושה את זה אני עם הפונקציות שרשמתי למעלה 
    unreserve(1, request);

    return slot;
}

//============================================================================================================

PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
    const std::vector<std::string>& promptTokens) const {
    const size_t tokensPerBlock = blockSize / embeddingSize;
    PrefixMatch returnVal = PrefixMatch{0, {}};
    if (promptTokens.size() < tokensPerBlock) {
        return returnVal;
    }
    
    // מתחיל מ0 עד לא כולל אחרון
    size_t prefixLength = tokensPerBlock;
    bool found = true;
    while (prefixLength <= promptTokens.size() && found) {

        std::vector<std::string> currentPrefix( promptTokens.begin(),
            promptTokens.begin() + prefixLength
        );

        found = false;

        for (std::size_t i = 0; i < cacheMetadata.size() && !found; i=i+1) {
            const CacheEntry& entry = cacheMetadata[i];

            if (entry.hasPrefix && entry.modelId == modelId &&
                entry.prefixTokens == currentPrefix) {

                returnVal.blockIds.push_back(i);
                returnVal.matchedTokens = prefixLength;
                found = true;
            }
        }
        prefixLength += tokensPerBlock;
    }

    return returnVal;
}