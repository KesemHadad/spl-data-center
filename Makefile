all: bin/server
 
bin/server: bin/Main.o bin/Model.o bin/Server.o
	@echo 'Linking'
	g++ -g -o bin/server bin/Main.o bin/Model.o bin/Server.o bin/Request.o bin/EmbeddingRequest.o bin/GenerativeRequest.o bin/ArrivalOrderPolicy.o bin/LongestPrefixPolicy.o bin/ShortestJobPolicy.o
    
bin/Main.o: src/Main.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/Main.o src/Main.cpp
    
bin/Model.o: src/Model.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/Model.o src/Model.cpp
 
bin/Server.o: src/Server.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/Server.o src/Server.cpp

bin/Request.o: src/Request.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/Request.o src/Request.cpp

bin/EmbeddingRequest.o: src/EmbeddingRequest.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/EmbeddingRequest.o src/EmbeddingRequest.cpp

bin/GenerativeRequest.o: src/GenerativeRequest.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/GenerativeRequest.o src/GenerativeRequest.cpp

bin/ArrivalOrderPolicy.o: src/ArrivalOrderPolicy.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/ArrivalOrderPolicy.o src/ArrivalOrderPolicy.cpp

bin/LongestPrefixPolicy.o: src/LongestPrefixPolicy.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/LongestPrefixPolicy.o src/LongestPrefixPolicy.cpp

bin/ShortestJobPolicy.o: src/ShortestJobPolicy.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/ShortestJobPolicy.o src/ShortestJobPolicy.cpp

.PHONY: clean
 
clean:
	@echo 'Cleaning'
	rm -f bin/*.o bin/server
