#ifndef SERVER_H_
#define SERVER_H_

#include <vector>
#include <string>
#include "Request.h"
#include "GPU.h"

struct GPUConfig{
    std::size_t memoryBytes;
};

struct ModelConfig{
    std::string id;
    std::size_t embeddingSize;
};

enum class RequestType { Generative, TextEmbedder };

struct RequestConfig{
    std::size_t id;
    RequestType type;
    std::string modelId;
    int admissionCycle;
    std::string prompt;
    std::string scriptedOutput;
    size_t maxOutputTokens;
};

struct Config{
    std::size_t blockSize;
    std::vector<GPUConfig> gpus;
    std::vector<ModelConfig> models;
    std::vector<RequestConfig> requests;
    std::string policy;
};

class Model;
class SchedulingPolicy;

class Server{
private:
   
    std::vector<GPU> gpus;
    std::vector<Model*> models;
    std::vector<Request*> nonSubmittedRequests;
    std::vector<Request*> waitingRequests;
    std::vector<Request*> activeRequests;
    SchedulingPolicy* policy;
    
    int cycle;
    std::vector<std::size_t> activeRequestGpus;
    std::size_t maxGpuBlocks;

    Config readConfig(std::string configPath) const;
    void clear();
    const Model& findModelById(const std::string& modelId) const;
    void copyFrom(const Server& other);
    void submitArrivingRequests();
    void admitWaitingRequests();
    void executeActiveRequests();
    int findBestGpuForRequest(const Request* req) const;
    void copyLongerPrefixIfAvailable(const Request* req, std::size_t chosenGpu);   

public:
    Server(std::string configPath);
    bool step();
    void serve(); 
    ~Server();
    Server(const Server& other);
    Server& operator=(const Server& other);
    Server(Server&& other);
    Server& operator=(Server&& other);
};

#endif