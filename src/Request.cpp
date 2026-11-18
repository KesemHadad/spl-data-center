#include "Request.h"
#include "GPU.h"
#include "Model.h"
  
    
Request::Request(int id, const Model& model, int admissionCycle, std::string prompt):
    id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)), reservedBlocks(0) {}
const Model& Request::getModel() const{ return model;}
int Request::getId() const{return id;}
int Request::getAdmissionCycle() const{return admissionCycle;}

void Request::deallocateMemory(GPU& gpu){
    for (std::size_t slot : memoryBlocks) {
        gpu.release(slot);
    }
    memoryBlocks.clear();

    if (reservedBlocks > 0) {
        gpu.unreserve(reservedBlocks, *this);
        reservedBlocks = 0;
    }
    }

