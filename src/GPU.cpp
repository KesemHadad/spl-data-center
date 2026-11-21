#include "GPU.h"
#include "MemoryBlock.h"



// Constructor
GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr) {

}

// Get Block Size
std::size_t GPU::getBlockSize() const { return this->blockSize; }

// Get Blocks For Memory
std::size_t GPU::getBlocksForMemory(std::size_t memory) const {
    if (memory % blockSize != 0)
        return (memory/blockSize) + 1;
    return memory/blockSize;
}

// Assumes BlockId, memory[BlockID] are valid and offset < blockSize
int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value ) {
    memory[blockId]->writeToBlock(offset, value);
}