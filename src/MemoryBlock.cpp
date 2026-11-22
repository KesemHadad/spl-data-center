#include "MemoryBlock.h"

// Constructor
MemoryBlock::MemoryBlock(size_t size)
    : size(size), refCount(0), data(new int[size]()) {
} // כתיבה בצורה כזו עם רשימת אתחול מונעת אתחול כפול משמע חוסכת זמן ריצה מיותר

// Read - מניח offset הוא אינדקס תקין
int MemoryBlock::readFromBlock(std::size_t offset) const { return data[offset]; }

// Write - מניח offset הוא אינדקס תקין
void MemoryBlock::writeToBlock(std::size_t offset, int value) { data[offset] = value; }

// Acquire
void MemoryBlock::acquire() { refCount = refCount + 1; }

// Release
IdleAction MemoryBlock::release() {
    refCount = refCount - 1;
    if (refCount == 0) { 
        return onLastRelease(); 
    }
    return IdleAction::KeepCached;
}



// Destructor
MemoryBlock::~MemoryBlock() { delete[] data; }

// Copy Constructor
MemoryBlock::MemoryBlock(const MemoryBlock& other)
    : size(other.size), refCount(0), data(new int[size]) {
        for (size_t i = 0; i<other.size; i=i+1) {
            data[i] = other.data[i];
        }
}

// Copy Assignment - סדר פעולות לפי עיקרון Strong Exception Guarantee
MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {

    if (this != &other) {
        int* newData = new int[other.size];

        for (size_t i = 0; i < other.size; i=i+1) {
            newData[i] = other.data[i];
        }

        delete[] this->data;
        this->refCount = 0;
        this->size = other.size;
        this->data = newData;
    }

    return *this;
}

// Move Constructor
MemoryBlock::MemoryBlock(MemoryBlock&& other) noexcept
    : size(other.size), refCount(other.refCount), data(other.data) {
        other.data = nullptr;
        other.size = 0;
        other.refCount =0;
} // מאפסים גם 2 השדות לשם שימור מצב עקבי, ניתן להתשמש בפונקציה לא רק על משתנים זמניים

// Move Assignment
MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) noexcept {

    if (this != &other) {

        delete[] this->data;
        this->refCount = other.refCount;
        this->size = other.size;
        this->data = other.data;

        other.data = nullptr;
        other.size = 0;
        other.refCount =0;

    }

    return *this;
}

int MemoryBlock::getRefCount() const { return refCount; }



KVBlock::KVBlock(size_t size) : MemoryBlock(size) { }
IdleAction KVBlock::onLastRelease() const { return IdleAction::KeepCached; }

MemoryBlock* KVBlock::clone() const {
    MemoryBlock* returnVal = new KVBlock(*this);
    return returnVal;
}


EmbeddingWorkspace::EmbeddingWorkspace(size_t size) : MemoryBlock(size) { }
IdleAction EmbeddingWorkspace::onLastRelease() const { return IdleAction::Destroy; }

MemoryBlock* EmbeddingWorkspace::clone() const {
    MemoryBlock* returnVal = new EmbeddingWorkspace(*this);
    return returnVal;
}

