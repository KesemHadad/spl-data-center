all: bin/server
 
bin/server: bin/Main.o bin/Model.o bin/Server.o
	@echo 'Linking'
	g++ -g -o bin/server bin/Main.o bin/Model.o bin/Server.o
    
bin/Main.o: src/Main.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/Main.o src/Main.cpp
    
bin/Model.o: src/Model.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/Model.o src/Model.cpp
 
bin/Server.o: src/Server.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude -o bin/Server.o src/Server.cpp


.PHONY: clean
 
clean:
	@echo 'Cleaning'
	rm -f bin/*.o bin/server
