#include "Request.h"
#include "GPU.h"
#include "Model.h"





EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt):
    Request(id, model, admissionCycle, prompt), output(""), isFinished(false), processedTokens(0) {}

std::vector<std::string> EmbeddingRequest::getCacheableTokens() const{ return {};}
bool EmbeddingRequest::finished() const{return isFinished;}
std::string EmbeddingRequest::getOutput() const{return output;}
void EmbeddingRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const{gpu.addMemoryBlock(*this, slot);}

Request* EmbeddingRequest::clone(const Model& model) const {
    EmbeddingRequest* copy = new EmbeddingRequest(id, model, admissionCycle,"");
    copy->tokens = tokens;
    copy-> memoryBlocks = memoryBlocks;
    copy -> reservedBlocks = reservedBlocks;
    copy-> output = output;
    copy -> isFinished = isFinished;
    copy -> processedTokens = processedTokens;
    return copy;
}

std::size_t EmbeddingRequest::getMaxNeededMemory() const{return (model.getEmbeddingSize() * 2);}

std::size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const{getMaxNeededMemory();}

std::size_t EmbeddingRequest::getNeedToUnreserve() const{return reservedBlocks;}

void EmbeddingRequest::reserveMemory(GPU& gpu) {
    std::size_t blocksNeeded = gpu.getBlocksForMemory(getNeededMemory(gpu));
    gpu.reserve(blocksNeeded, *this);
    reservedBlocks = blocksNeeded;
    for (std::size_t i = 0; i < blocksNeeded; i++) {
        std::size_t slot = gpu.allocateBlock(*this);
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
        reservedBlocks--;
    }
}


void EmbeddingRequest::act(GPU& gpu) {
    if (isFinished) {
        return;
    }
    if (tokens.empty()) {
        isFinished = true;
        return;
    }

    std::size_t e = model.getEmbeddingSize();
    std::size_t sumSlot = memoryBlocks[0];
    std::size_t currSlot;
    std::size_t currOffset;

    if (memoryBlocks.size() == 1) {
        currSlot = memoryBlocks[0];
        currOffset = e;
    } else {
        currSlot = memoryBlocks[1];
        currOffset = 0;
    }

    std::vector<std::string> currentToken = {tokens[processedTokens]};
    model.generateTokenEmbedding(currentToken, gpu, currSlot, currOffset);

    for (std::size_t i = 0; i < e; i++) {
        int sumVal = gpu.readFromBlock(sumSlot, i);
        int tokVal = gpu.readFromBlock(currSlot, currOffset + i);
        gpu.writeToBlock(sumSlot, i, sumVal + tokVal);
    }

    processedTokens++;

    if (processedTokens == tokens.size()) {
        output = "[";
        for (std::size_t i = 0; i < e; i++) {
            int sumVal = gpu.readFromBlock(sumSlot, i);
            int avg = sumVal / static_cast<int>(tokens.size());
            output += std::to_string(avg);
            if (i + 1 < e) {
                output += ", ";
            }
        }
        output += "]";
        isFinished = true;
    }
}
