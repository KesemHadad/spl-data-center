#include <fstream>
#include <stdexcept>
#include "json.hpp"
#include <iostream>
#include "Server.h"
#include "Model.h"
#include "SchedulingPolicy.h"
#include <utility>


using namespace std;
using json = nlohmann::json;

Server::Server(std::string configPath): 
    gpus(),
    models(),
    nonSubmittedRequests(),
    waitingRequests(),
    activeRequests(),
    policy(nullptr),
    cycle(0),
    activeRequestGpus(),
    maxGpuBlocks(0) {

    Config cfg = readConfig(configPath);

    for (std::size_t i = 0; i < cfg.gpus.size(); i++) {
        std::size_t blocksNum = cfg.gpus[i].memoryBytes / cfg.blockSize;
        gpus.push_back(GPU(blocksNum, cfg.blockSize));
        if (blocksNum > maxGpuBlocks) {
            maxGpuBlocks = blocksNum;
        }
    }

    for (std::size_t i = 0; i < cfg.models.size(); i++) {
        models.push_back(new Model(cfg.models[i].id, cfg.models[i].embeddingSize));
    }

    for (std::size_t i = 0; i < cfg.requests.size(); i++) {
        const RequestConfig& reqCfg = cfg.requests[i];

        Model* modelPtr = nullptr;
        for (std::size_t m = 0; m < models.size(); m++) {
            if (models[m]->getModelId() == reqCfg.modelId) {
                modelPtr = models[m];
                break;
            }
        }
        if (modelPtr == nullptr) {
            clear();
            throw std::invalid_argument("unknown model id");
        }

        if (reqCfg.type == RequestType::Generative) {
            nonSubmittedRequests.push_back(new GenerativeRequest(
                static_cast<int>(reqCfg.id),
                *modelPtr,
                reqCfg.admissionCycle,
                reqCfg.prompt,
                reqCfg.scriptedOutput,
                reqCfg.maxOutputTokens
            ));
        } else {
            nonSubmittedRequests.push_back(new EmbeddingRequest(
                static_cast<int>(reqCfg.id),
                *modelPtr,
                reqCfg.admissionCycle,
                reqCfg.prompt
            ));
        }
    }

    if (cfg.policy == "arrivalOrder") {
        policy = new ArrivalOrderPolicy();
    } else if (cfg.policy == "shortestJob") {
        policy = new ShortestJobPolicy();
    } else if (cfg.policy == "longestPrefix") {
        policy = new LongestPrefixPolicy();
    } else {
        clear();
        throw std::invalid_argument("invalid scheduling policy");
    }
}







Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

void Server::serve() {
    while (step()) {
    }
    std::size_t totalCacheHits = 0;
    std::size_t totalEvictions = 0;
    for (std::size_t i = 0; i < gpus.size(); i++) {
        totalCacheHits += gpus[i].getCacheHits();
        totalEvictions += gpus[i].getEvictions();
    }

    std::cout << "summary: cycles=" << cycle
            << " cache_hits=" << totalCacheHits
            << " evictions=" << totalEvictions << std::endl;
}

void Server::clear() {
    for (std::size_t i = 0; i < nonSubmittedRequests.size(); i++) {
        delete nonSubmittedRequests[i];
    }
    nonSubmittedRequests.clear();

    for (std::size_t i = 0; i < waitingRequests.size(); i++) {
        delete waitingRequests[i];
    }
    waitingRequests.clear();

    for (std::size_t i = 0; i < activeRequests.size(); i++) {
        delete activeRequests[i];
    }
    activeRequests.clear();

    for (std::size_t i = 0; i < models.size(); i++) {
        delete models[i];
    }
    models.clear();

    delete policy;
    policy = nullptr;

    gpus.clear();
    activeRequestGpus.clear();
    cycle = 0;
    maxGpuBlocks = 0;
}

const Model& Server::findModelById(const std::string& modelId) const {
    for (std::size_t i = 0; i < models.size(); i++) {
        if (models[i]->getModelId() == modelId) {
            return *models[i];
        }
    }
    throw std::runtime_error("model not found");
}

void Server::copyFrom(const Server& other) {
    gpus = other.gpus;
    cycle = other.cycle;
    activeRequestGpus = other.activeRequestGpus;
    maxGpuBlocks = other.maxGpuBlocks;

    for (std::size_t i = 0; i < other.models.size(); i++) {
        models.push_back(new Model(*other.models[i]));
    }

    for (std::size_t i = 0; i < other.nonSubmittedRequests.size(); i++) {
        const Model& copiedModel = findModelById(other.nonSubmittedRequests[i]->getModel().getModelId());
        nonSubmittedRequests.push_back(other.nonSubmittedRequests[i]->clone(copiedModel));
    }

    for (std::size_t i = 0; i < other.waitingRequests.size(); i++) {
        const Model& copiedModel = findModelById(other.waitingRequests[i]->getModel().getModelId());
        waitingRequests.push_back(other.waitingRequests[i]->clone(copiedModel));
    }

    for (std::size_t i = 0; i < other.activeRequests.size(); i++) {
        const Model& copiedModel = findModelById(other.activeRequests[i]->getModel().getModelId());
        activeRequests.push_back(other.activeRequests[i]->clone(copiedModel));
    }

    if (other.policy != nullptr) {
        policy = other.policy->clone();
    } else {
        policy = nullptr;
    }
}

Server::~Server() {
    clear();
}

Server::Server(const Server& other)
    : gpus(),
      models(),
      nonSubmittedRequests(),
      waitingRequests(),
      activeRequests(),
      policy(nullptr),
      cycle(0),
      activeRequestGpus(),
      maxGpuBlocks(0) {
    copyFrom(other);
}

Server& Server::operator=(const Server& other) {
    if (this != &other) {
        clear();
        copyFrom(other);
    }
    return *this;
}

Server::Server(Server&& other)
    : gpus(std::move(other.gpus)),
      models(std::move(other.models)),
      nonSubmittedRequests(std::move(other.nonSubmittedRequests)),
      waitingRequests(std::move(other.waitingRequests)),
      activeRequests(std::move(other.activeRequests)),
      policy(other.policy),
      cycle(other.cycle),
      activeRequestGpus(std::move(other.activeRequestGpus)),
      maxGpuBlocks(other.maxGpuBlocks) {
    other.gpus.clear();
    other.models.clear();
    other.nonSubmittedRequests.clear();
    other.waitingRequests.clear();
    other.activeRequests.clear();
    other.policy = nullptr;
    other.cycle = 0;
    other.activeRequestGpus.clear();
    other.maxGpuBlocks = 0;
}

Server& Server::operator=(Server&& other) {
    if (this != &other) {
        clear();
        gpus = std::move(other.gpus);
        models = std::move(other.models);
        nonSubmittedRequests = std::move(other.nonSubmittedRequests);
        waitingRequests = std::move(other.waitingRequests);
        activeRequests = std::move(other.activeRequests);
        policy = other.policy;
        cycle = other.cycle;
        activeRequestGpus = std::move(other.activeRequestGpus);
        maxGpuBlocks = other.maxGpuBlocks;

        other.gpus.clear();
        other.models.clear();
        other.nonSubmittedRequests.clear();
        other.waitingRequests.clear();
        other.activeRequests.clear();
        other.policy = nullptr;
        other.cycle = 0;
        other.activeRequestGpus.clear();
        other.maxGpuBlocks = 0;
    }
    return *this;
}

bool Server::step() {
    if (nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty()) {
        return false;
    }

    submitArrivingRequests();
    admitWaitingRequests();
    executeActiveRequests();

    cycle++;
    return true;
}

void Server::submitArrivingRequests() {
    std::vector<Request*> remaining;

    for (std::size_t i = 0; i < nonSubmittedRequests.size(); i++) {
        Request* req = nonSubmittedRequests[i];

        if (req->getAdmissionCycle() == cycle) {
            bool canEverFit = !gpus.empty() &&
                              gpus[0].getBlocksForMemory(req->getMaxNeededMemory()) <= maxGpuBlocks;
            if (!canEverFit) {
                std::cout << "rejected " << req->getId()
                          << " cycle=" << cycle << std::endl;
                delete req;
            } else {
                waitingRequests.push_back(req);
            }
        } else {
            remaining.push_back(req);
        }
    }

    nonSubmittedRequests = remaining;
}

void Server::executeActiveRequests() {
    std::vector<Request*> remainingRequests;
    std::vector<std::size_t> remainingGpus;

    for (std::size_t i = 0; i < activeRequests.size(); i++) {
        Request* req = activeRequests[i];
        std::size_t gpuIdx = activeRequestGpus[i];

        req->act(gpus[gpuIdx]);

        if (req->finished()) {
            std::cout << "finished " << req->getId()
                      << " cycle=" << cycle
                      << " gpu=" << gpuIdx << ": "
                      << req->getOutput() << std::endl;
            req->deallocateMemory(gpus[gpuIdx]);
            delete req;
        } else {
            remainingRequests.push_back(req);
            remainingGpus.push_back(gpuIdx);
        }
    }

    activeRequests = remainingRequests;
    activeRequestGpus = remainingGpus;
}

void Server::admitWaitingRequests() {
    if (waitingRequests.empty()) {
        return;
    }

    std::vector<const Request*> eligible;
    for (std::size_t i = 0; i < waitingRequests.size(); i++) {
        eligible.push_back(waitingRequests[i]);
    }
    std::vector<int> order = policy->choose(eligible, gpus);

    std::vector<bool> admitted(waitingRequests.size(), false);

    for (std::size_t k = 0; k < order.size(); k++) {
        std::size_t reqIdx = static_cast<std::size_t>(order[k]);
        Request* req = waitingRequests[reqIdx];

        int chosenGpu = findBestGpuForRequest(req);
        if (chosenGpu == -1) {
            continue;
        }

        std::size_t gpuIdx = static_cast<std::size_t>(chosenGpu);
        copyLongerPrefixIfAvailable(req, gpuIdx);

        req->reserveMemory(gpus[gpuIdx]);
        activeRequests.push_back(req);
        activeRequestGpus.push_back(gpuIdx);
        admitted[reqIdx] = true;
    }

    std::vector<Request*> remaining;
    for (std::size_t i = 0; i < waitingRequests.size(); i++) {
        if (!admitted[i]) {
            remaining.push_back(waitingRequests[i]);
        }
    }
    waitingRequests = remaining;
}

void Server::copyLongerPrefixIfAvailable(const Request* req, std::size_t chosenGpu) {
    std::vector<std::string> cacheable = req->getCacheableTokens();
    if (cacheable.empty()) {
        return;
    }

    const std::string& modelId = req->getModel().getModelId();
    std::size_t embSize = req->getModel().getEmbeddingSize();

    PrefixMatch chosenMatch = gpus[chosenGpu].findLongestCachedPrefix(modelId, embSize, cacheable);
    std::size_t longestBlocks = chosenMatch.blockIds.size();
    int sourceGpu = -1;
    std::vector<std::size_t> sourceBlocks;

    for (std::size_t g = 0; g < gpus.size(); g++) {
        if (g == chosenGpu) {
            continue;
        }
        PrefixMatch otherMatch = gpus[g].findLongestCachedPrefix(modelId, embSize, cacheable);
        if (otherMatch.blockIds.size() > longestBlocks) {
            longestBlocks = otherMatch.blockIds.size();
            sourceGpu = static_cast<int>(g);
            sourceBlocks = otherMatch.blockIds;
        }
    }

    if (sourceGpu != -1) {
        gpus[chosenGpu].copyMemoryFromOtherGPU(gpus[sourceGpu], sourceBlocks);
    }
}

int Server::findBestGpuForRequest(const Request* req) const {
    std::vector<std::string> cacheable = req->getCacheableTokens();
    const std::string& modelId = req->getModel().getModelId();
    std::size_t embSize = req->getModel().getEmbeddingSize();

    int bestGpu = -1;
    std::size_t bestCachedBlocks = 0;
    std::size_t bestNeededSlots = 0;

    for (std::size_t g = 0; g < gpus.size(); g++) {
        std::size_t cachedBlocks = 0;
        std::size_t idleCachedBlocks = 0;

        if (!cacheable.empty()) {
            PrefixMatch match = gpus[g].findLongestCachedPrefix(modelId, embSize, cacheable);
            cachedBlocks = match.blockIds.size();
            for (std::size_t b = 0; b < match.blockIds.size(); b++) {
                if (gpus[g].isSlotIdle(match.blockIds[b])) {
                    idleCachedBlocks++;
                }
            }
        }

        std::size_t newBlocks = gpus[g].getBlocksForMemory(req->getNeededMemory(gpus[g]));
        std::size_t neededSlots = newBlocks + idleCachedBlocks;

        if (neededSlots <= gpus[g].availableSlots()) {
            bool isBetter = (bestGpu == -1) ||
                            (cachedBlocks > bestCachedBlocks) ||
                            (cachedBlocks == bestCachedBlocks && neededSlots < bestNeededSlots);
            if (isBetter) {
                bestGpu = static_cast<int>(g);
                bestCachedBlocks = cachedBlocks;
                bestNeededSlots = neededSlots;
            }
        }
    }

    return bestGpu;
}

