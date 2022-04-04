#include "problemGenerator.h"
#include <exception>

int main(int argc, char* argv[])
{
	char* algoName = argv[1];
	unsigned int uiProblemSeed = atoi(argv[2]); // used for generating the random COP problem
	COP GCop(uiProblemSeed);
}