
image:image.c image.h
	gcc -g -fopenmp image.c -o image -lm
clean:
	rm -f image output.png