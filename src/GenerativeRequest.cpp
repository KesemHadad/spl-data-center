#include "Request.h"
#include "GPU.h"
#include "Model.h"


GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt,
     std::string scriptedOutput,std::size_t maxOutputTokens):
     Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens),
     processedTokens(0),outputTokens(model.tokenizer(scriptedOutput)), isFinished(false), output("") {
        if (outputTokens.size() > maxOutputTokens) {
            outputTokens.resize(maxOutputTokens);
        }
     }

std::vector<std::string> GenerativeRequest::getCacheableTokens() const {return tokens;}
void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {gpu.addMemoryBlock(*this, slot);}
std::size_t GenerativeRequest::getNeedToUnreserve() const {return reservedBlocks;}
std::size_t GenerativeRequest::getMaxNeededMemory() const {return model.getEmbeddingSize() * (tokens.size() + maxOutputTokens);}


std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const {
    PrefixMatch match = gpu.findLongestCachedPrefix(model.getModelId(), model.getEmbeddingSize(), tokens);
    return model.getEmbeddingSize() * (tokens.size() - match.matchedTokens + maxOutputTokens);
}

bool GenerativeRequest::finished() const {return isFinished;}

std::string GenerativeRequest::getOutput() const {return output;}

Request* GenerativeRequest::clone(const Model& model) const {
        GenerativeRequest* copy = new GenerativeRequest(id, model, admissionCycle,"", scriptedOutput, maxOutputTokens);
    copy -> processedTokens = processedTokens;
    copy->tokens = tokens;
    copy-> memoryBlocks = memoryBlocks;
    copy -> reservedBlocks = reservedBlocks;
    copy-> output = output;
    copy -> isFinished = isFinished;
    return copy;
}

void GenerativeRequest::reserveMemory(GPU& gpu) {
    PrefixMatch match = gpu.findLongestCachedPrefix(model.getModelId(), model.getEmbeddingSize(), tokens);
    for(std::size_t slot : match.blockIds) {
    gpu.acquire(slot);
    memoryBlocks.push_back(slot);
    }
    processedTokens = match.matchedTokens;
    std::size_t remainingMemory = model.getEmbeddingSize() * (tokens.size() - match.matchedTokens + maxOutputTokens);
    std::size_t blocksToReserve = gpu.getBlocksForMemory(remainingMemory);
    if (blocksToReserve > 0) {
        gpu.reserve(blocksToReserve, *this);
        reservedBlocks = blocksToReserve;
    }
}

void GenerativeRequest::act(GPU& gpu) {
    if (isFinished) {
        return;
    }
    if (processedTokens >= tokens.size() + outputTokens.size()) {
        isFinished = true;
        return;
    }

    std::size_t e = model.getEmbeddingSize();
    std::size_t tokensPerBlock = gpu.getBlockSize() / e;
    std::size_t blockIdx = processedTokens / tokensPerBlock;
    std::size_t tokenInBlock = processedTokens % tokensPerBlock;
    std::size_t offset = tokenInBlock * e;

    if (blockIdx == memoryBlocks.size()) {
        std::size_t slot = gpu.allocateBlock(*this);
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
        reservedBlocks--;
    }

    std::size_t currSlot = memoryBlocks[blockIdx];

    std::vector<std::string> sequence;
    if (processedTokens < tokens.size()) {
        for (std::size_t i = 0; i <= processedTokens; i++) {
            sequence.push_back(tokens[i]);
        }
    } else {
        sequence = tokens;
        std::size_t outIdx = processedTokens - tokens.size();
        for (std::size_t i = 0; i <= outIdx; i++) {
            sequence.push_back(outputTokens[i]);
        }
    }

    model.generateTokenEmbedding(sequence, gpu, currSlot, offset);

    if (processedTokens >= tokens.size()) {
        std::size_t outIdx = processedTokens - tokens.size();
        if (outIdx > 0) {
            output += " ";
        }
        output += outputTokens[outIdx];
    }

    processedTokens++;

    if (processedTokens % tokensPerBlock == 0) {
        gpu.assignPrefixToBlock(currSlot, sequence, model.getModelId());
    }

    if (processedTokens == tokens.size() + outputTokens.size()) {
        isFinished = true;
    }
    
}
