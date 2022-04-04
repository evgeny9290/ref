#pragma once
// Warning : The use of static variables makes this code unsuitable for parallel run
//           The fix is not complex: use function arguments for outputs instead of return value, or create an assignment operator to create a copy of a complex type, not a pointer to it 

//#include "stdafx.h"
#include <stdio.h>      /* printf, scanf, puts, NULL */
//#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <memory.h>     /* memset */
#include <math.h>       /* exp */
#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <assert.h>
#include <algorithm>	/* copy */
#include <chrono>
#include <random>
#include <string>
#include "Windows.h"

#define SaveValues
//#define SaveSolutions
//#define CheckClocks
//#define Couts

#if defined SaveValues || defined SaveSolutions
const unsigned int SAVE_FREQ = 1;
#endif

using namespace std;

std::string GetCurrentDirectory()
{
	char buffer[MAX_PATH];
	GetModuleFileNameA(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");

	return std::string(buffer).substr(0, pos);
}

#pragma region Constraint Optimization Problem
class COP
{
public:
	static const string m_strCOPDirectory;
	//consts
	static const unsigned int MAX_VAR_PRIORITIES = 20; // must link to VALUES_RANGE[] array somehow. what is variable priority? Condition Variables prio?
	static const unsigned long MAX_TOTAL_VALUES = 6000; // upper ceiling of the problem (for matrix of constrains)
	static const unsigned long MAX_NUM_OF_VARS = 200; // maximum number of Condition Variables
	static const unsigned long MAX_VALUES_OF_VAR = 200; // maximum domain for every Condition Variable
	static const unsigned char MAX_NUM_OF_MS = 20; // number of types of resources
	static const unsigned char MAX_NUM_OF_R_PER_M = 30; // max number of resourse per action/tactic for each type
	static const unsigned long MAX_LENGTH_OF_GRADES_VECTOR = 19; // number of objectives in object func.
	static const unsigned char LEVEL_OF_Q = 4; // constrain priority lvl 4. the lower the better
	static const unsigned char LEVEL_OF_BINARY_CONSTRAINTS = 3; // constrain priority lvl 3
	static const unsigned char LEVEL_OF_B = 2; // constrain priority lvl 2
	static const unsigned char PRIORITIES_NUM = 1; // constrain priority lvl 1
	static const unsigned char MAX_CONSTRAINTS_RATIO = 10; // 
	static const unsigned char NUM_OF_P_VALUES = 10; // image domain is of size 10
	static const unsigned char NUM_OF_Q_VALUES = 10; // image domain is of size 10
	static const unsigned char MAX_NUM_OF_ELITE = 10; // Elite size for genetic algorithm
	//typedefs
	typedef struct VarData // Condition Variable Data?
	{
		//unsigned int uiID;
		//unsigned int uiIndex;
		unsigned char ucPrio; // priority of this specific Condition Variable?
		//unsigned char ucClass;
		//unsigned int auiValuesIndex[MAX_VALUES_OF_VAR];
		unsigned char aucValuesQ[MAX_VALUES_OF_VAR]; // f_q...
		bool abValuesB[MAX_VALUES_OF_VAR]; // f_b...
		unsigned char aucValuesP[MAX_VALUES_OF_VAR];
		unsigned char aucValuesM[MAX_VALUES_OF_VAR];
		unsigned long ulValuesAmount; // store "size" i.e number of VarData

	} VarData;

	typedef struct ValuesPerVars
	{
		VarData VarsData[MAX_NUM_OF_VARS];
		unsigned int uiValidVarAmount; // store "size" i.e number of ValuesPerVars
		unsigned int uiVarPrioAmount[MAX_VAR_PRIORITIES]; // isnt used anywhere? 
	} ValuesPerVars;

	typedef struct M // resource type
	{
		unsigned char ucAmount; // amount of that type
	} M;

	typedef struct SolutionVector
	{
		unsigned long aulSolutionVector[MAX_NUM_OF_VARS]; //[X_i = j, ...]
	} SolutionVector;

	typedef struct GradesVector //  for every solution vector there is a grade vector
	{							// becuase of lexicographic evaluation of constrains
		static const unsigned long VALUES_RANGE[MAX_LENGTH_OF_GRADES_VECTOR];
		float afGradesVector[MAX_LENGTH_OF_GRADES_VECTOR];
		double Scalarization() const;
	} GradesVector;

	//Members
	ValuesPerVars* ValuesPerVariables;
	unsigned char* aaucBinaryConstraintsMatrix;
	unsigned int uiMaxValuesNum;
	M Ms[MAX_NUM_OF_MS];

	//Random Engine for random problem creation
	std::mt19937 ReproducibleRandomEngine;  // the Mersenne Twister engine

	//Methods
	COP();
	COP(unsigned int uiProblemSeed);

	~COP();

	void operator=(COP& Cop);
	void GenerateSingleNeighbor(SolutionVector& OutputNeighbor, const SolutionVector& CurrentSolution, unsigned int uiNumOfVarChangesInNeighborhood, std::mt19937& AlgorithmRandomEngine);
	void EvaluateSolution(const SolutionVector& Solution, GradesVector& Evaluation);
	inline int BinConsIdx(int iRow, int iCol) { return iRow * MAX_TOTAL_VALUES + iCol; };
	void WorstSolution(SolutionVector& Output);
	void WorstValue(GradesVector& Output);

	ofstream m_ofValuesPerVariables; // my addition
	ofstream m_ofBinaryConstraintsMatrix; // my addition
	ofstream m_ofMs; // my addition

};

//const string COP::m_strCOPDirectory = "C:\\Users\\evgni\\Desktop\\Projects\\LocalSearch\\LocalSearch\\Problems\\";
//const string COP::m_strCOPDirectory = "C:\\Users\\evgni\\Desktop\\projects_mine\\ref\\ref\\LocalSearchProblemGenerator\\LocalSearchProblemGenerator\\CPP_Problems\\";
 const string COP::m_strCOPDirectory = GetCurrentDirectory() + "/" + "../../copsimpleai/CPP_Problems/";

//Random Engine for algorithm behaviour
std::mt19937 AlgorithmRandomEngine;

inline int CompareVarByPrio(const void* VarA, const void* VarB)
{

	COP::VarData* pVarA = (COP::VarData*)VarA;
	COP::VarData* pVarB = (COP::VarData*)VarB;

	if (pVarA->ucPrio > pVarB->ucPrio) return 1;
	if (pVarA->ucPrio < pVarB->ucPrio) return -1;

	std::uniform_int_distribution<uint32_t> UIntDistForTieBreak(0, 1);
	return (UIntDistForTieBreak(AlgorithmRandomEngine) == 0) ? 1 : -1;
}

//Ranges of GradesVector (Depends on EvaluateSolution)  
const unsigned long COP::GradesVector::VALUES_RANGE[] = { COP::MAX_NUM_OF_VARS * 1,
														 COP::MAX_NUM_OF_VARS * 10,
														 COP::MAX_NUM_OF_VARS * 1,
														 COP::MAX_NUM_OF_VARS * 10,
														 COP::MAX_NUM_OF_VARS * 1,
														 COP::MAX_NUM_OF_VARS * 10,
														 COP::MAX_NUM_OF_VARS * 1,
														 COP::MAX_NUM_OF_VARS * 10,
														 COP::MAX_NUM_OF_VARS * 1,
														 COP::MAX_NUM_OF_VARS * 10,
														 COP::MAX_NUM_OF_VARS * 1,
														 COP::MAX_NUM_OF_VARS * 10,
														 COP::MAX_NUM_OF_VARS * 1,
														 COP::MAX_NUM_OF_VARS * 10,
														 COP::MAX_NUM_OF_VARS * 1,
														 COP::MAX_NUM_OF_VARS * 10,
														 COP::MAX_NUM_OF_VARS * 1,
														 COP::MAX_CONSTRAINTS_RATIO * (COP::MAX_NUM_OF_VARS * COP::MAX_NUM_OF_VARS - COP::MAX_NUM_OF_VARS) / 2,
														 COP::MAX_NUM_OF_VARS * 1
}; // weights i.e constant factors for each Condition Variable? i.e its priority?
   // must link to MAX_VAR_PRIORITIES somehow

double COP::GradesVector::Scalarization() const // sum of util's with respect to weights? 
{
	//todo: Add overflow defense .. changed to lower case in order to push into git
	double dScalarValue = 0;
	double dCurrentWeight = 1;

	for (int iGradeIdx = MAX_LENGTH_OF_GRADES_VECTOR - 1; iGradeIdx >= 0; iGradeIdx--)
	{
		if (VALUES_RANGE[iGradeIdx] == 0) continue;

		dScalarValue += afGradesVector[iGradeIdx] * dCurrentWeight;
		dCurrentWeight *= (double)(VALUES_RANGE[iGradeIdx] + 1);
	}

	return dScalarValue;
}


COP::COP()
{
	ValuesPerVariables = new ValuesPerVars;
	aaucBinaryConstraintsMatrix = new unsigned char[COP::MAX_TOTAL_VALUES * COP::MAX_TOTAL_VALUES]; // x * num_cols + y? 2d matrix implement on 1d array?
}

COP::COP(unsigned int uiProblemSeed)
{
	m_ofValuesPerVariables.open(m_strCOPDirectory + "ValuesPerVariable_" + to_string(uiProblemSeed) + ".csv"); // my addition
	m_ofBinaryConstraintsMatrix.open(m_strCOPDirectory + "BinaryConstraintsMatrix_" + to_string(uiProblemSeed) + ".txt"); // my addition
	m_ofMs.open(m_strCOPDirectory + "Ms_" + to_string(uiProblemSeed) + ".txt"); // my addition

	//Same as COP() - Written again since it might be used within older C++ than C++11
	ValuesPerVariables = new ValuesPerVars;
	aaucBinaryConstraintsMatrix = new unsigned char[COP::MAX_TOTAL_VALUES * COP::MAX_TOTAL_VALUES]; // x * num_cols + y? 2d matrix implement on 1d array?
	///

	//Seeding by uiProblemSeed for reproducible problems
	ReproducibleRandomEngine.seed(uiProblemSeed);

	std::uniform_int_distribution<uint32_t> UIntDistForVars(1, COP::MAX_NUM_OF_VARS);
	std::uniform_int_distribution<uint32_t> UIntDistForConstraintRatio(2, COP::MAX_CONSTRAINTS_RATIO);
	std::uniform_int_distribution<uint32_t> UIntDistForMsAmount(0, COP::MAX_NUM_OF_R_PER_M);
	std::uniform_int_distribution<uint32_t> UIntDistForPrio(0, COP::PRIORITIES_NUM - 1); // which prio
	std::uniform_int_distribution<uint32_t> UIntDistForB(0, 1);
	std::uniform_int_distribution<uint32_t> UIntDistForM(0, COP::MAX_NUM_OF_MS - 1);
	std::uniform_int_distribution<uint32_t> UIntDistForP(1, COP::NUM_OF_P_VALUES);
	std::uniform_int_distribution<uint32_t> UIntDistForQ(1, COP::NUM_OF_Q_VALUES);

	unsigned int uiConstraintsRatio = UIntDistForConstraintRatio(ReproducibleRandomEngine);

	//Control constraint ratio by problem seed 
	//if (uiProblemSeed >= 1000 && uiProblemSeed <=2000) uiConstraintsRatio = 1;


	std::uniform_int_distribution<uint32_t> UIntDistForConstraint(0, uiConstraintsRatio);

	const unsigned int uiVarNum = UIntDistForVars(ReproducibleRandomEngine);
	uiMaxValuesNum = min(COP::MAX_TOTAL_VALUES / uiVarNum, COP::MAX_VALUES_OF_VAR);

	//Fill Ms data
	for (unsigned int m = 0; m < COP::MAX_NUM_OF_MS; m++)
	{
		Ms[m].ucAmount = UIntDistForMsAmount(ReproducibleRandomEngine);
		m_ofMs << to_string(Ms[m].ucAmount) << " ";
		/*m_ofMs << to_string(Ms[m].ucAmount) << " ";*/ // my addition
		/*if (uiProblemSeed >= 1000 && uiProblemSeed <=2000)
		{
			if (m%5==0) Ms[m].ucAmount = 1;
			else Ms[m].ucAmount = 0;
		}*/

	}

	m_ofMs << to_string(uiMaxValuesNum) << " "; // my addition
	m_ofMs << to_string(uiVarNum); // my addition
	//Fill ValuesPerVariables Data
	ValuesPerVariables->uiValidVarAmount = uiVarNum;
	m_ofValuesPerVariables << "index" << "," << "B" << "," << "M" << "," << "P" << "," << "Q" << "," << "ulValuesAmount" << "," << "ucPrio" << endl; // my addition
	for (unsigned int variable = 0; variable < uiVarNum; variable++)
	{
		ValuesPerVariables->VarsData[variable].ulValuesAmount = uiMaxValuesNum;
		ValuesPerVariables->VarsData[variable].ucPrio = UIntDistForPrio(ReproducibleRandomEngine);

		for (unsigned int value = 0; value < uiMaxValuesNum; value++)
		{
			ValuesPerVariables->VarsData[variable].abValuesB[value] = (UIntDistForB(ReproducibleRandomEngine) == 0) ? false : true;
			ValuesPerVariables->VarsData[variable].aucValuesM[value] = UIntDistForM(ReproducibleRandomEngine);
			ValuesPerVariables->VarsData[variable].aucValuesP[value] = UIntDistForP(ReproducibleRandomEngine);
			ValuesPerVariables->VarsData[variable].aucValuesQ[value] = UIntDistForQ(ReproducibleRandomEngine);
			m_ofValuesPerVariables << to_string(variable) << "," << to_string(ValuesPerVariables->VarsData[variable].abValuesB[value]) << ","; // my addition
			m_ofValuesPerVariables << to_string(ValuesPerVariables->VarsData[variable].aucValuesM[value]) << ","; // my addition
			m_ofValuesPerVariables << to_string(ValuesPerVariables->VarsData[variable].aucValuesP[value]) << ","; // my addition
			m_ofValuesPerVariables << to_string(ValuesPerVariables->VarsData[variable].aucValuesQ[value]) << ","; // my addition
			m_ofValuesPerVariables << to_string(ValuesPerVariables->VarsData[variable].ulValuesAmount) << ","; // my addition
			m_ofValuesPerVariables << to_string(ValuesPerVariables->VarsData[variable].ucPrio) << endl; // my addition

		}
	}


	//Fill Constraints Data - memcpy would be faster 
	for (unsigned int variable1 = 0; variable1 < ValuesPerVariables->uiValidVarAmount; variable1++) {
		for (unsigned int value1 = 0; value1 < ValuesPerVariables->VarsData[variable1].ulValuesAmount; value1++) {
			for (unsigned int variable2 = 0; variable2 < ValuesPerVariables->uiValidVarAmount; variable2++) {
				for (unsigned int value2 = 0; value2 < ValuesPerVariables->VarsData[variable2].ulValuesAmount; value2++)
				{
					assert(variable1 * uiMaxValuesNum + value1 < COP::MAX_TOTAL_VALUES&&
						variable2* uiMaxValuesNum + value2 < COP::MAX_TOTAL_VALUES); // func below maps to "4d" array? out of bounds?
					int val = UIntDistForConstraint(ReproducibleRandomEngine); // my addition
					aaucBinaryConstraintsMatrix[BinConsIdx(variable1 * uiMaxValuesNum + value1, variable2 * uiMaxValuesNum + value2)] = val;
					// my addition
					if (variable1 == ValuesPerVariables->uiValidVarAmount - 1 &&
						value1 == ValuesPerVariables->VarsData[variable1].ulValuesAmount - 1 &&
						variable2 == ValuesPerVariables->uiValidVarAmount - 1 &&
						value2 == ValuesPerVariables->VarsData[variable2].ulValuesAmount - 1) {
						m_ofBinaryConstraintsMatrix << to_string(val);
					}

					else {
						m_ofBinaryConstraintsMatrix << to_string(val) << " ";
					}
				}
			}
		}
	}

}

COP::~COP()
{
	delete ValuesPerVariables;
	delete[] aaucBinaryConstraintsMatrix;
	m_ofBinaryConstraintsMatrix.close(); // my addition
	m_ofMs.close(); // my addition
	m_ofValuesPerVariables.close(); // my addition

}