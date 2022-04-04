#pragma once
// Warning : The use of static variables makes this code unsuitable for parallel run
//           The fix is not complex: use function arguments for outputs instead of return value, or create an assignment operator to create a copy of a complex type, not a pointer to it 

#include "stdafx.h"
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
		unsigned char *aaucBinaryConstraintsMatrix;
		unsigned int uiMaxValuesNum;
		M Ms[MAX_NUM_OF_MS];

		//Random Engine for random problem creation
		std::mt19937 ReproducibleRandomEngine;  // the Mersenne Twister engine

		//Methods
		COP();
		COP(unsigned int uiProblemSeed);

		~COP();

		void operator=(COP& Cop);
		void GenerateSingleNeighbor(SolutionVector& OutputNeighbor, const SolutionVector& CurrentSolution,unsigned int uiNumOfVarChangesInNeighborhood, std::mt19937& AlgorithmRandomEngine);
		void EvaluateSolution(const SolutionVector& Solution, GradesVector& Evaluation);
		inline int BinConsIdx(int iRow, int iCol) {return iRow*MAX_TOTAL_VALUES+iCol;};
		void WorstSolution(SolutionVector& Output);
		void WorstValue(GradesVector& Output);

		ofstream m_ofValuesPerVariables; // my addition
		ofstream m_ofBinaryConstraintsMatrix; // my addition
		ofstream m_ofMs; // my addition
		
};

//Random Engine for algorithm behaviour
std::mt19937 AlgorithmRandomEngine;

inline int CompareVarByPrio (const void * VarA, const void * VarB)
{
  
  COP::VarData* pVarA = (COP::VarData *)VarA;
  COP::VarData* pVarB = (COP::VarData *)VarB;

  if (pVarA->ucPrio > pVarB->ucPrio) return 1;
  if (pVarA->ucPrio < pVarB->ucPrio) return -1;

  std::uniform_int_distribution<uint32_t> UIntDistForTieBreak(0,1);
  return (UIntDistForTieBreak(AlgorithmRandomEngine) == 0)? 1: -1;
}

//Ranges of GradesVector (Depends on EvaluateSolution)  
const unsigned long COP::GradesVector::VALUES_RANGE[] = {COP::MAX_NUM_OF_VARS*1,
													     COP::MAX_NUM_OF_VARS*10,
													     COP::MAX_NUM_OF_VARS*1,
													     COP::MAX_NUM_OF_VARS*10,
													     COP::MAX_NUM_OF_VARS*1,
													     COP::MAX_NUM_OF_VARS*10,
													     COP::MAX_NUM_OF_VARS*1,
													     COP::MAX_NUM_OF_VARS*10,
													     COP::MAX_NUM_OF_VARS*1,
														 COP::MAX_NUM_OF_VARS*10,
														 COP::MAX_NUM_OF_VARS*1,
														 COP::MAX_NUM_OF_VARS*10,
														 COP::MAX_NUM_OF_VARS*1,
														 COP::MAX_NUM_OF_VARS*10,
														 COP::MAX_NUM_OF_VARS*1,
														 COP::MAX_NUM_OF_VARS*10,
														 COP::MAX_NUM_OF_VARS*1,
														 COP::MAX_CONSTRAINTS_RATIO*(COP::MAX_NUM_OF_VARS*COP::MAX_NUM_OF_VARS-COP::MAX_NUM_OF_VARS)/2,
														 COP::MAX_NUM_OF_VARS*1
													   }; // weights i.e constant factors for each Condition Variable? i.e its priority?
														  // must link to MAX_VAR_PRIORITIES somehow

double COP::GradesVector::Scalarization() const // sum of util's with respect to weights? 
{
	//todo: Add overflow defense .. changed to lower case in order to push into git
	double dScalarValue = 0;
	double dCurrentWeight = 1;

	for (int iGradeIdx = MAX_LENGTH_OF_GRADES_VECTOR - 1; iGradeIdx>=0;iGradeIdx--)
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
	aaucBinaryConstraintsMatrix = new unsigned char[COP::MAX_TOTAL_VALUES*COP::MAX_TOTAL_VALUES]; // x * num_cols + y? 2d matrix implement on 1d array?
}

COP::COP(unsigned int uiProblemSeed)
{

	//Same as COP() - Written again since it might be used within older C++ than C++11
	ValuesPerVariables = new ValuesPerVars;
	aaucBinaryConstraintsMatrix = new unsigned char[COP::MAX_TOTAL_VALUES*COP::MAX_TOTAL_VALUES]; // x * num_cols + y? 2d matrix implement on 1d array?
	///

	//Seeding by uiProblemSeed for reproducible problems
	ReproducibleRandomEngine.seed(uiProblemSeed);
	
	std::uniform_int_distribution<uint32_t> UIntDistForVars(1,COP::MAX_NUM_OF_VARS);
	std::uniform_int_distribution<uint32_t> UIntDistForConstraintRatio(2,COP::MAX_CONSTRAINTS_RATIO);
	std::uniform_int_distribution<uint32_t> UIntDistForMsAmount(0,COP::MAX_NUM_OF_R_PER_M);
	std::uniform_int_distribution<uint32_t> UIntDistForPrio(0,COP::PRIORITIES_NUM-1); // which prio
	std::uniform_int_distribution<uint32_t> UIntDistForB(0,1);
	std::uniform_int_distribution<uint32_t> UIntDistForM(0,COP::MAX_NUM_OF_MS-1);
	std::uniform_int_distribution<uint32_t> UIntDistForP(1,COP::NUM_OF_P_VALUES);
	std::uniform_int_distribution<uint32_t> UIntDistForQ(1,COP::NUM_OF_Q_VALUES);

	unsigned int uiConstraintsRatio = UIntDistForConstraintRatio(ReproducibleRandomEngine);
	
	//Control constraint ratio by problem seed 
	//if (uiProblemSeed >= 1000 && uiProblemSeed <=2000) uiConstraintsRatio = 1;
	

	std::uniform_int_distribution<uint32_t> UIntDistForConstraint(0,uiConstraintsRatio);

	const unsigned int uiVarNum = UIntDistForVars(ReproducibleRandomEngine);
	uiMaxValuesNum = min(COP::MAX_TOTAL_VALUES/uiVarNum, COP::MAX_VALUES_OF_VAR);

	//Fill Ms data
	for (unsigned int m=0; m<COP::MAX_NUM_OF_MS; m++)
	{
		Ms[m].ucAmount = UIntDistForMsAmount(ReproducibleRandomEngine);
		/*if (uiProblemSeed >= 1000 && uiProblemSeed <=2000) 
		{
			if (m%5==0) Ms[m].ucAmount = 1;
			else Ms[m].ucAmount = 0;
		}*/
			
	}

	//Fill ValuesPerVariables Data
	ValuesPerVariables->uiValidVarAmount = uiVarNum;
	for (unsigned int variable=0; variable<uiVarNum; variable++)
	{
		ValuesPerVariables->VarsData[variable].ulValuesAmount = uiMaxValuesNum;
		ValuesPerVariables->VarsData[variable].ucPrio = UIntDistForPrio(ReproducibleRandomEngine);

		for (unsigned int value=0; value<uiMaxValuesNum; value++)
		{
				ValuesPerVariables->VarsData[variable].abValuesB[value] =  (UIntDistForB(ReproducibleRandomEngine) == 0)? false:true;
				ValuesPerVariables->VarsData[variable].aucValuesM[value] = UIntDistForM(ReproducibleRandomEngine);
				ValuesPerVariables->VarsData[variable].aucValuesP[value] = UIntDistForP(ReproducibleRandomEngine);
				ValuesPerVariables->VarsData[variable].aucValuesQ[value] = UIntDistForQ(ReproducibleRandomEngine);

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
				}
			}
		}
	}

}

COP::~COP() 
{
	delete ValuesPerVariables;
	delete [] aaucBinaryConstraintsMatrix;
}

void COP::operator=(COP& Cop)
{
	
	*(ValuesPerVariables) = *(Cop.ValuesPerVariables);
	uiMaxValuesNum = COP::MAX_TOTAL_VALUES/Cop.ValuesPerVariables->uiValidVarAmount; // what is this calculation?

	std::copy(&(Cop.aaucBinaryConstraintsMatrix[0]), &(Cop.aaucBinaryConstraintsMatrix[0])+ MAX_TOTAL_VALUES*MAX_TOTAL_VALUES,&(aaucBinaryConstraintsMatrix[0]));
	
	//memcpy(&(aaucBinaryConstraintsMatrix[0]),&(Cop.aaucBinaryConstraintsMatrix[0]),MAX_TOTAL_VALUES*MAX_TOTAL_VALUES); // why noy memcpy? is it different?

	for (unsigned int m=0; m<COP::MAX_NUM_OF_MS; m++)
		Ms[m] = Cop.Ms[m];
}

void COP::GenerateSingleNeighbor(COP::SolutionVector& OutputNeighbor, const COP::SolutionVector& CurrentSolution,unsigned int uiNumOfVarChangesInNeighborhood, std::mt19937& AlgorithmRandomEngine)
{
	std::uniform_int_distribution<uint32_t> UIntDistForVariable(0,ValuesPerVariables->uiValidVarAmount);
	
	for (unsigned int uiIndex=0; uiIndex < COP::MAX_NUM_OF_VARS; uiIndex++)
	{
		OutputNeighbor.aulSolutionVector[uiIndex] = CurrentSolution.aulSolutionVector[uiIndex]; // just to keep the currentSolution intact
	}

	for (unsigned int uiVar=1; uiVar<=uiNumOfVarChangesInNeighborhood; uiVar++) // input i.e X_i = k where X_i random variable get an input of k
	{
		unsigned long ulRandomVariable = UIntDistForVariable(AlgorithmRandomEngine);
		std::uniform_int_distribution<uint32_t> UIntDistForValue(0,ValuesPerVariables->VarsData[ulRandomVariable].ulValuesAmount);
		unsigned long ulRandomValue = UIntDistForValue(AlgorithmRandomEngine);

		OutputNeighbor.aulSolutionVector[ulRandomVariable] = ulRandomValue;
	} //  here we will have a neighbor with "uiNumOfVarChangesInNeighborhood" different inputs than the CurrentSolution
}

void COP::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation ) // one of the most important funcs (imo)
{
	//static GradesVector Evaluation;
	memset(&Evaluation.afGradesVector,0,sizeof(float)*MAX_LENGTH_OF_GRADES_VECTOR);
	//memset(&Evaluation,0,sizeof(GradesVector));

	M MsUsage[MAX_NUM_OF_MS]; // here will be the output of the current resource vector? while Ms is the resouce vector until now
	memcpy(MsUsage,Ms,sizeof(M)*MAX_NUM_OF_MS);
	
	//Compute Evaluation from Solution
	for (unsigned long ulCurrentSolutionVar = 0; ulCurrentSolutionVar < ValuesPerVariables->uiValidVarAmount; ulCurrentSolutionVar++)
	{
		
		bool bCurrentIsLegal = true;
		unsigned long ulCurrentValue = Solution.aulSolutionVector[ulCurrentSolutionVar];
		unsigned char ucCurrentM = ValuesPerVariables->VarsData[ulCurrentSolutionVar].aucValuesM[ulCurrentValue];
		
		//Check for enoungh resources, if not - skip this variable in the evaluation (but it stays in the solution)
		if (MsUsage[ucCurrentM].ucAmount == 0) 
			continue;

		unsigned char ucCurrentPrio = ValuesPerVariables->VarsData[ulCurrentSolutionVar].ucPrio;
		bool bCurrentB = ValuesPerVariables->VarsData[ulCurrentSolutionVar].abValuesB[ulCurrentValue];
		unsigned char ucCurrentQ = ValuesPerVariables->VarsData[ulCurrentSolutionVar].aucValuesQ[ulCurrentValue];
		unsigned char ucCurrentP = ValuesPerVariables->VarsData[ulCurrentSolutionVar].aucValuesP[ulCurrentValue];
		
		//Assumption: Evaluation is done in the order of priorities so partial solution achieved so far is the best we can achieve
		//            without changing the order of vars in the same priority. Amount of m is zero ==> Solution Cannot be extended

		
		for(unsigned long ulPastSolutionVar = 0; ulPastSolutionVar < ulCurrentSolutionVar; ulPastSolutionVar++)
		{
			unsigned long ulPastValue = Solution.aulSolutionVector[ulPastSolutionVar];

			assert(ulCurrentSolutionVar*uiMaxValuesNum+ulCurrentValue <= MAX_TOTAL_VALUES &&
				   ulPastSolutionVar*uiMaxValuesNum+ulPastValue <= MAX_TOTAL_VALUES);

			unsigned char ucCurrentBinaryValue = aaucBinaryConstraintsMatrix[BinConsIdx(ulCurrentSolutionVar*uiMaxValuesNum+ulCurrentValue, ulPastSolutionVar*uiMaxValuesNum+ulPastValue)];

			if (ucCurrentBinaryValue == 0)
			{
				bCurrentIsLegal = false;
				break;
			}
			Evaluation.afGradesVector[LEVEL_OF_BINARY_CONSTRAINTS] -= ucCurrentBinaryValue;
		}

		//Infeasable parts are not contributing to the GradesVector
		if (bCurrentIsLegal)
		{
			//-= : minimization problem
			MsUsage[ucCurrentM].ucAmount-=1;
			Evaluation.afGradesVector[2*ucCurrentPrio] -= 1;
			Evaluation.afGradesVector[2*ucCurrentPrio+1] -= ucCurrentP;
			Evaluation.afGradesVector[COP::LEVEL_OF_B] -= bCurrentB;
			Evaluation.afGradesVector[COP::LEVEL_OF_Q] -= ucCurrentQ;
		}
	}

	return;
}


ostream& operator<<(ostream& os, const COP::GradesVector& GV) // "print" grade vector after summing according to weights
{

	for (unsigned char i=0; i<COP::MAX_LENGTH_OF_GRADES_VECTOR; i++)
	{
		os << GV.afGradesVector[i] << " ";
	}
	os << GV.Scalarization();
    
	return os;
}

ofstream& operator<<(ofstream& ofs, const COP::GradesVector& GV) // "print" grade vector after summing according to weights
{

	for (unsigned char i=0; i<COP::MAX_LENGTH_OF_GRADES_VECTOR; i++)
	{
		ofs << GV.afGradesVector[i] << " ";
	}
    ofs << GV.Scalarization();

	return ofs;
}

ostream& operator<<(ostream& os, const COP::SolutionVector& SV)
{

	for (unsigned char i=0; i<COP::MAX_NUM_OF_VARS; i++)
	{
		os << SV.aulSolutionVector[i] << " ";
	}
    
	return os;
}

ofstream& operator<<(ofstream& ofs, const COP::SolutionVector& SV)
{

	for (unsigned char i=0; i<COP::MAX_NUM_OF_VARS; i++)
	{
		ofs << SV.aulSolutionVector[i] << " ";
	}
    
	return ofs;
}

inline bool operator< (const COP::GradesVector& lhs, const COP::GradesVector& rhs) // lexicographical evaluation 
{ 
	for (unsigned long ulGrade = 0; ulGrade < COP::MAX_LENGTH_OF_GRADES_VECTOR; ulGrade++)
	{
		if (lhs.afGradesVector[ulGrade] < rhs.afGradesVector[ulGrade])
			return true;
		
		if (lhs.afGradesVector[ulGrade] > rhs.afGradesVector[ulGrade])
			return false;
	}

	return false;
}

inline bool operator<= (const COP::GradesVector& lhs, const COP::GradesVector& rhs) // lexicographical evaluation 
{
	for (unsigned long ulGrade = 0; ulGrade < COP::MAX_LENGTH_OF_GRADES_VECTOR; ulGrade++)
	{	
		if (lhs.afGradesVector[ulGrade] < rhs.afGradesVector[ulGrade])
			return true;

		if (lhs.afGradesVector[ulGrade] > rhs.afGradesVector[ulGrade])
			return false;
	}

	return true;

}
inline bool operator> (const COP::GradesVector& lhs, const COP::GradesVector& rhs) // lexicographical evaluation 
{ 
	return rhs < lhs;
}


float operator-(const COP::GradesVector  &num1, const COP::GradesVector &num2) // if difference is greater than the "percision" available, return that difference else returh whatever u can?
{
	float fDiff = 0;
	for (unsigned int uiGrade=0; uiGrade < COP::MAX_LENGTH_OF_GRADES_VECTOR; uiGrade++)
	{
		float fDiff = (num1.afGradesVector[uiGrade] - num2.afGradesVector[uiGrade])/(uiGrade+1);
		if (fabs(fDiff)>FLT_EPSILON)
			return fDiff;
	}

    return -fDiff;
}
//iiii
COP::GradesVector operator-(const COP::GradesVector& num1, float num2) // my addition (for Great Deluge Algorithm, calculation m_Beta)
{
	static COP::GradesVector DeltaSolution;
	for (unsigned int uiGrade = 0; uiGrade < COP::MAX_LENGTH_OF_GRADES_VECTOR; uiGrade++)
	{
		DeltaSolution.afGradesVector[uiGrade] = num1.afGradesVector[uiGrade] - num2;
	}

	return DeltaSolution;
}

COP::SolutionVector operator-(const COP::SolutionVector  &Sol1, const COP::SolutionVector &Sol2) // difference between two solutions for each entry
{
	static COP::SolutionVector DeltaSolution;
	for (unsigned int uiSolIndex=0; uiSolIndex < COP::MAX_NUM_OF_VARS; uiSolIndex++)
		DeltaSolution.aulSolutionVector[uiSolIndex] = Sol1.aulSolutionVector[uiSolIndex]-Sol2.aulSolutionVector[uiSolIndex];

    return DeltaSolution;
}

COP::SolutionVector operator+(const COP::SolutionVector& Sol1, const COP::SolutionVector& Sol2) // adding two solutions for each entry // my addition
{
	static COP::SolutionVector DeltaSolution;
	for (unsigned int uiSolIndex = 0; uiSolIndex < COP::MAX_NUM_OF_VARS; uiSolIndex++)
		DeltaSolution.aulSolutionVector[uiSolIndex] = Sol1.aulSolutionVector[uiSolIndex] + Sol2.aulSolutionVector[uiSolIndex];

	return DeltaSolution;
}

void COP::WorstValue(GradesVector& Output)
{
	for (unsigned short usGrade=0; usGrade<MAX_LENGTH_OF_GRADES_VECTOR; usGrade++)
		Output.afGradesVector[usGrade] = 0;
}

void COP::WorstSolution(SolutionVector& Output)
{
	for (unsigned int uiSolIndex=0; uiSolIndex < COP::MAX_NUM_OF_VARS; uiSolIndex++)
		Output.aulSolutionVector[uiSolIndex] = 0;
}
#pragma endregion

#pragma region Time Measure

struct HighResClock
{
        typedef long long                               rep;
        typedef std::nano                               period;
        typedef std::chrono::duration<rep, period>      duration;
        typedef std::chrono::time_point<HighResClock>   time_point;
        static const bool is_steady = true;

        static time_point now();
};

namespace
{
    const long long g_Frequency = []() -> long long  // g_Frequency get initialized via lambda func? (create func and execute same line) [update clock frequency]
    {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        return frequency.QuadPart;
    }();
}

HighResClock::time_point HighResClock::now()
{
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return time_point(duration(count.QuadPart * static_cast<rep>(period::den) / g_Frequency));
}
#pragma endregion

#pragma region CFixedTimeSearch
///
/// An abstract CFixedTimeSearch class - A general local search
/// For a specific local search implementation:
/// 1. Define the two template typenames : SolutionType and ValueType
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    c. AcceptanceCritirionReached() 
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///
template<typename SolutionType, typename ValueType> 
class CFixedTimeSearch 
{
public:

	static const unsigned int MAX_TIME_MEASURES = 18000;
	static const string m_strDirectory;

	CFixedTimeSearch(const string strPostFileName);
	~CFixedTimeSearch();

	//Entry point for the algorithm
	virtual double Solve(const SolutionType& InitialSolution, 
						 double dTimeoutInSeconds, 
						 unsigned long ulMaxIterations,
						 unsigned long ulSeed);
	virtual void GetBestSolution(SolutionType& BestSolution);
	virtual void GetBestValue(ValueType& BestValue);


	//Solution methods
	virtual void ChooseCandidate(SolutionType& ChosenNeighbor) = 0;
	virtual void CountedEvaluateSolution(const SolutionType& Solution, ValueType& Evaluation);
	virtual void EvaluateSolution(const SolutionType& Solution, ValueType& Evaluation) = 0;
	virtual bool AcceptanceCritirionReached(const SolutionType& Solution) = 0;
	virtual bool StopCritirionReached() = 0;
	virtual bool RestartCritirionReached() = 0;
	virtual void Restart(SolutionType& NewInitial) = 0; 

	//Display final results
	void FinalResultsReport(double dCurrentTime);

	//Time measure methods
	void StartClock();
	double CheckClock(int iMeasurePosition);
	double ConditionalCheckClock(int iMeasurePosition);
	void TimesReport();

	//Solutions Recording methods
	void MemorizeSolutionAndValue(double dTime);
	void WriteSolutionsAndValues();

	//Solution members
	SolutionType m_CurrentSolution; // [X_i = k,...] for every X_i Condition variable we evaluate the optimal input?
	SolutionType m_BestSolution; // [X_i = k,...] for every X_i Condition variable we evaluate the optimal input?
	ValueType m_ValueOfBestSolution; // grades vec of solution. i.e eval_solution([X_i = k,...])
 	ValueType m_ValueOfCurrentSolution; // inputs for every Condition variable?
	unsigned long m_ulIterations; // current iteration number
	unsigned long m_ulMaxIterations; // max iterations possible

	//Solution Recording members
	double *m_dTimes;

#ifdef SaveSolutions
	SolutionType *m_ArrayOfCurrentSolutions;
	SolutionType *m_ArrayOfBestSolutions;
#endif

#ifdef SaveValues
	ValueType *m_ArrayOfCurrentValues;
	ValueType *m_ArrayOfBestValues;
#endif

	unsigned int m_uiSolutionIdx;

	//File writing members
	ofstream m_ofTimes;

#ifdef SaveSolutions
	ofstream m_ofCurrentSolution;
	ofstream m_ofBestSolution;
#endif

#ifdef SaveValues
	ofstream m_ofCurrentValue;
	ofstream m_ofBestValue;
#endif

	//Time measure members
	//chrono::time_point< chrono::high_resolution_clock, chrono::nanoseconds > m_StartWallTime;
	HighResClock::time_point m_StartWallTime;

	static const string m_sMeasureStrings[]; 
	typedef enum E_MEASURE_NAMES 
	{
		AFTER_CHOOSE_NEIGHBOR=0,
		AFTER_EVALUATE_SOLUTION,
		AFTER_ACCEPT_CRITIRION_REACHED,
		AFTER_MEMORIZING,
		AFTER_RESTART_CRITIRION_REACHED,
		A_MUST_CHECK_CLOCK
	} E_MEASURE_NAMES;
	double *m_dMeasureTime;
	E_MEASURE_NAMES* m_eMeasurePositions; //[MAX_TIME_MEASURES];
	unsigned int m_uiMeasureIdx;

};

template<typename SolutionType, typename ValueType> 
CFixedTimeSearch<SolutionType, ValueType>::CFixedTimeSearch(string strPostFileName) // opening file to save the data basically
{
	m_ofTimes.open(m_strDirectory + "Times_" + strPostFileName + ".txt"); // why doesnt create new file output?

#ifdef SaveSolutions
	//m_ofCurrentSolution.open(m_strDirectory + "CurrentSolution_" + strPostFileName + ".txt");
	//m_ofBestSolution.open(m_strDirectory + "BestSolution_" + strPostFileName + ".txt");
#endif

#ifdef SaveValues
	m_ofCurrentValue.open(m_strDirectory + "CurrentValue_" + strPostFileName + ".txt"); // why doesnt create new file output?
	m_ofBestValue.open(m_strDirectory + "BestValue_" + strPostFileName + ".txt"); // why doesnt create new file output?
#endif

	m_dTimes = new double[MAX_ITERATIONS];

#ifdef SaveSolutions
	m_ArrayOfCurrentSolutions = new SolutionType[MAX_ITERATIONS];
	m_ArrayOfBestSolutions = new SolutionType[MAX_ITERATIONS];
#endif

#ifdef SaveValues
	m_ArrayOfCurrentValues = new ValueType[MAX_ITERATIONS];
	m_ArrayOfBestValues = new ValueType[MAX_ITERATIONS];
#endif

	m_dMeasureTime = new double[MAX_TIME_MEASURES];
	m_eMeasurePositions = new E_MEASURE_NAMES[MAX_TIME_MEASURES];
}

template<typename SolutionType, typename ValueType> 
CFixedTimeSearch<SolutionType, ValueType>::~CFixedTimeSearch() // closing all needed files basically and cleaning whats allocated
{
	m_ofTimes.close();

#ifdef SaveSolutions
	m_ofCurrentSolution.close();
	m_ofBestSolution.close();
#endif SaveSolutions
	
#ifdef SaveValues
	m_ofCurrentValue.close();
	m_ofBestValue.close();
#endif

	delete[] m_dTimes;

#ifdef SaveSolutions
	delete[] m_ArrayOfCurrentSolutions;
	delete[] m_ArrayOfBestSolutions;
#endif

#ifdef SaveValues
	delete[] m_ArrayOfCurrentValues;
	delete[] m_ArrayOfBestValues;
#endif

	delete[] m_dMeasureTime;
}

template<typename SolutionType, typename ValueType> 
const string CFixedTimeSearch<SolutionType, ValueType>::m_strDirectory = GetCurrentDirectory() + "/" + "../../copsimpleai/LocalSearch/Results/";
//const string CFixedTimeSearch<SolutionType, ValueType>::m_strDirectory = "C:\\Users\\evgni\\Desktop\\projects_mine\\ref\\ref\\Projects\\LocalSearch\\LocalSearch\Results\\";
//const string CFixedTimeSearch<SolutionType, ValueType>::m_strDirectory = "C:\\Users\\evgni\\Desktop\\projects_mine\\ref\\ref\\Projects\\LocalSearch\\LocalSearch\Results\\";


template<typename SolutionType, typename ValueType> 
const string CFixedTimeSearch<SolutionType, ValueType>::m_sMeasureStrings[6] = {"After ChooseCandidate",
																			"After EvaluateSolution",
																			"After AcceptanceCritirionReached",
																			"After Memorizing",
																			"After RestartCritirionReached",
																			"A Must CheckClock"};

///
///Function EvaluateSolution
///
template<typename SolutionType, typename ValueType> 
void CFixedTimeSearch<SolutionType, ValueType>::StartClock()
{
	m_StartWallTime = HighResClock::now();
	m_uiMeasureIdx = 0;
}

///
///Function EvaluateSolution
///
template<typename SolutionType, typename ValueType> 
double CFixedTimeSearch<SolutionType, ValueType>::CheckClock(int iMeasurePosition)
{
	HighResClock::time_point now = HighResClock::now();
	double dTimeDiff = chrono::duration <double, nano>(now - m_StartWallTime).count();

	if (m_uiMeasureIdx < MAX_TIME_MEASURES)
	{
		m_dMeasureTime[m_uiMeasureIdx] = dTimeDiff;
		m_eMeasurePositions[m_uiMeasureIdx] = (E_MEASURE_NAMES)iMeasurePosition;
	}
	m_uiMeasureIdx++;
	
	/*if (m_uiMeasureIdx==MAX_TIME_MEASURES)
		cout << " Time measures out of memory ";*/

	return dTimeDiff/1000000000;
}

template<typename SolutionType, typename ValueType> 
double CFixedTimeSearch<SolutionType, ValueType>::ConditionalCheckClock(int iMeasurePosition)
{
#ifdef CheckClocks
	return CheckClock(iMeasurePosition);
#else
	if (iMeasurePosition == A_MUST_CHECK_CLOCK || iMeasurePosition == AFTER_ACCEPT_CRITIRION_REACHED)
		return CheckClock(iMeasurePosition);
	else
		return 0.0;
#endif
}

template<typename SolutionType, typename ValueType> 
void CFixedTimeSearch<SolutionType, ValueType>::MemorizeSolutionAndValue(double dTime)
{
	m_dTimes[m_uiSolutionIdx] = dTime;

#ifdef SaveSolutions
	m_ArrayOfCurrentSolutions[m_uiSolutionIdx] = m_CurrentSolution;
	m_ArrayOfBestSolutions[m_uiSolutionIdx] = m_BestSolution;
#endif

#ifdef SaveValues
	m_ArrayOfCurrentValues[m_uiSolutionIdx] = m_ValueOfCurrentSolution;
	m_ArrayOfBestValues[m_uiSolutionIdx] = m_ValueOfBestSolution;
#endif

	m_uiSolutionIdx++;
}

template<typename SolutionType, typename ValueType> 
void CFixedTimeSearch<SolutionType, ValueType>::WriteSolutionsAndValues()
{
	for (unsigned int uiIdx = 1; uiIdx < m_uiSolutionIdx; uiIdx++)
	{
		if ((uiIdx % SAVE_FREQ) == 0) 
		{
			m_ofTimes << m_dTimes[uiIdx] << endl;

	#ifdef SaveSolutions
			m_ofCurrentSolution << m_ArrayOfCurrentSolutions[uiIdx] << endl;
			m_ofBestSolution << m_ArrayOfBestSolutions[uiIdx] << endl;
	#endif

	#ifdef SaveValues
			m_ofCurrentValue << m_ArrayOfCurrentValues[uiIdx] << endl; 
			m_ofBestValue << m_ArrayOfBestValues[uiIdx] << endl;
	#endif
		}
	}
}

template<typename SolutionType, typename ValueType> 
void CFixedTimeSearch<SolutionType, ValueType>::FinalResultsReport(double dCurrentTime)
{
	//cout << " Current Value: "  << m_ValueOfCurrentSolution;
	cout << " Best Value:" << m_ValueOfBestSolution << endl;
	cout << " Time: " << dCurrentTime << " Iterations: " << m_ulIterations << endl;
}

template<typename SolutionType, typename ValueType> 
void CFixedTimeSearch<SolutionType, ValueType>::TimesReport()
{
#ifdef CheckClocks
	/*cout << "Press Enter for CheckClock Report" << endl;
	getchar();
	for (unsigned int uiTimes = 0; uiTimes < m_uiMeasureIdx; uiTimes++)
	{
		cout << m_sMeasureStrings[m_eMeasurePositions[uiTimes]] << ": " << m_dMeasureTime[uiTimes] << endl;
	}*/
#endif
}

///
///Function Solve()
///
template<typename SolutionType, typename ValueType> 
double CFixedTimeSearch<SolutionType, ValueType>::Solve(const SolutionType& InitialSolution, 
												    double dTimeoutInSeconds, 
												    unsigned long ulMaxIterations,
												    unsigned long ulSeed) 
{
	
	StartClock();

	//Seeding by time for stochastic behaviour of algorithms
	AlgorithmRandomEngine.seed(ulSeed);

	m_uiSolutionIdx = 0;
	m_ulIterations = 0;
	m_ulMaxIterations = ulMaxIterations;
	m_CurrentSolution = m_BestSolution = InitialSolution;
	CountedEvaluateSolution(InitialSolution,m_ValueOfBestSolution);
	m_ValueOfCurrentSolution = m_ValueOfBestSolution;

	SolutionType CandidateSolution;
	ValueType ValueOfCandidateSolution; // only for evaluating solution
	double dCurrentTime = 0;

	do 
	{

		ChooseCandidate(CandidateSolution);

		ConditionalCheckClock(AFTER_CHOOSE_NEIGHBOR); 

		CountedEvaluateSolution(CandidateSolution, ValueOfCandidateSolution);
		
		ConditionalCheckClock(AFTER_EVALUATE_SOLUTION);

		if (AcceptanceCritirionReached(CandidateSolution))
		{
			m_CurrentSolution = CandidateSolution;
			m_ValueOfCurrentSolution = ValueOfCandidateSolution;

			//Minimization problem
			if (m_ValueOfCurrentSolution < m_ValueOfBestSolution)
			{
				m_BestSolution = m_CurrentSolution;
				m_ValueOfBestSolution = m_ValueOfCurrentSolution;
			}
		}

		dCurrentTime = ConditionalCheckClock(AFTER_ACCEPT_CRITIRION_REACHED);

		MemorizeSolutionAndValue(dCurrentTime);

		ConditionalCheckClock(AFTER_MEMORIZING);

		if (RestartCritirionReached())
			Restart(m_CurrentSolution);

		dCurrentTime = ConditionalCheckClock(A_MUST_CHECK_CLOCK);

	} 
	while (!StopCritirionReached() && (dCurrentTime < dTimeoutInSeconds));

	FinalResultsReport(dCurrentTime);
	TimesReport();
	WriteSolutionsAndValues();
	return dCurrentTime;
}

template<typename SolutionType, typename ValueType> 
void CFixedTimeSearch<SolutionType, ValueType>::GetBestSolution(SolutionType& BestSolution)
{
	BestSolution = m_BestSolution;
}

template<typename SolutionType, typename ValueType> 
void CFixedTimeSearch<SolutionType, ValueType>::GetBestValue(ValueType& BestValue)
{
	BestValue = m_ValueOfBestSolution;
}

template<typename SolutionType, typename ValueType> 
void CFixedTimeSearch<SolutionType, ValueType>::CountedEvaluateSolution(const SolutionType& Solution, ValueType& Evaluation)
{
	EvaluateSolution(Solution, Evaluation);
	m_ulIterations++;
}

#pragma endregion

#pragma region Abstract Specific Local Search
#pragma region CSimulatedAnnealingSearch
///
/// An abstract CSimulatedAnnealingSearch class - A general simulated annealing search, inherits from CFixedTimeSearch
/// For a specific simulated annealing search implementation:
/// 1. Define the two template typenames : SolutionType and ValueType
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///
template<typename SolutionType, typename ValueType> 
class CSimulatedAnnealingSearch : public CFixedTimeSearch <SolutionType, ValueType> 
{
public:
	
	CSimulatedAnnealingSearch(const string strPostFileName);

	virtual double Solve(const SolutionType& InitialSolution, 
					     double dTimeoutInSeconds, 
					     unsigned long ulMaxIterations,
					     float fInitialTemprature, 
					     float fTempratureStep,
					     unsigned long ulSeed);

	virtual void ChooseCandidate(SolutionType& Chosen)=0;
	virtual void EvaluateSolution(const SolutionType& Solution, ValueType& Evalution)=0;
	virtual bool AcceptanceCritirionReached(const SolutionType& Solution);
	virtual bool StopCritirionReached()=0;
	virtual bool RestartCritirionReached()=0;
	virtual void Restart(SolutionType& NewInitial)=0;
	float m_fTempratureStep;
	float m_fTemprature;

};

template<typename SolutionType, typename ValueType> 
CSimulatedAnnealingSearch<SolutionType, ValueType>::CSimulatedAnnealingSearch(const string strPostFileName)
	: CFixedTimeSearch(strPostFileName)
{
	
}

template<typename SolutionType, typename ValueType> 
double CSimulatedAnnealingSearch<SolutionType, ValueType>::Solve(const SolutionType& InitialSolution, 
																 double dTimeoutInSeconds, 
																 unsigned long ulMaxIterations,
																 float fInitialTemprature, 
																 float fTempratureStep,
																 unsigned long ulSeed)
{
	m_fTemprature = fInitialTemprature;
	m_fTempratureStep = fTempratureStep;

	return CFixedTimeSearch::Solve(InitialSolution, 
							   dTimeoutInSeconds, 
							   ulMaxIterations,
							   ulSeed);
}

template<typename SolutionType, typename ValueType> 
bool CSimulatedAnnealingSearch<SolutionType, ValueType>::AcceptanceCritirionReached(const SolutionType& Solution) 
{

	ValueType CandidateValue;
	CountedEvaluateSolution(Solution, CandidateValue);

	m_fTemprature = m_fTemprature/m_fTempratureStep;

	double Delta = (double)(CandidateValue - m_ValueOfCurrentSolution);

	//Negative Delta is an improvement in a minimization problem
	double dAcceptanceProbability = 0.0;
	if (Delta < 0)
		dAcceptanceProbability = 1.0;
	else 
		dAcceptanceProbability = (double)exp(-Delta / m_fTemprature);

#ifdef Couts
	cout << "Temprature: " << m_fTemprature << " AcceptaceP: " <<dAcceptanceProbability << endl;
#endif

	std::uniform_real_distribution<float> fDistribution(0.0,1.0);
	float fRand = fDistribution(AlgorithmRandomEngine);
	
	if (fRand<=dAcceptanceProbability)
		return true;

	return false;
}
#pragma endregion

#pragma region CTabuSearch
///
/// An abstract CTabuSearch class - A general tabu search, inherits from CFixedTimeSearch
/// For a specific tabu search implementation:
/// 1. Define the two template typenames : SolutionType and ValueType
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///
template<typename SolutionType, typename ValueType> 
class CTabuSearch : public CFixedTimeSearch <SolutionType, ValueType> 
{
public:
	
	static const unsigned long MAX_NEIGHBORS_COUNT = 10;
	static const unsigned long MAX_TABU_SIZE = 500;

	CTabuSearch(const string strPostFileName);

	virtual double Solve(const SolutionType& InitialSolution, 
						 double dTimeoutInSeconds, 
						 unsigned long ulMaxIterations,
						 unsigned long ulMaxTabuSize,
						 unsigned long ulSeed);

	virtual void ChooseCandidate(SolutionType& Chosen);
	virtual void GenerateNeighbors() = 0;
	virtual void EvaluateSolution(const SolutionType& Solution, ValueType& Evaluation)=0;
	virtual bool AcceptanceCritirionReached(const SolutionType& Solution);
	virtual bool StopCritirionReached()=0;
	virtual bool RestartCritirionReached()=0;
	virtual void Restart(SolutionType& NewInitial)=0; 
	virtual bool TabuContainsAttribute(float fAttribute);
	virtual void TabuPushAttribute(float fAttribute);
	virtual bool TabuForbidsSolution(const SolutionType& Solution)=0; //Pure virtual which should use TabuContainsAttributes
	virtual void TabuUpdate(const SolutionType& Solution)=0; //Pure Virtual which should use Tabu PushAttribute

	SolutionType m_CurrentNeighbors[MAX_NEIGHBORS_COUNT];
	unsigned long m_ulNeighborsCount;

	float m_fTabuList[MAX_TABU_SIZE];
	unsigned long m_ulMaxTabuSize;
	unsigned long m_ulTabuSize;
	unsigned long m_ulNextPushIndex;
};

template<typename SolutionType, typename ValueType> 
double CTabuSearch<SolutionType, ValueType>::Solve(const SolutionType& InitialSolution, 
												   double dTimeoutInSeconds, 
												   unsigned long ulMaxIterations,
												   unsigned long ulMaxTabuSize,
												   unsigned long ulSeed)
{
	m_ulMaxTabuSize = ulMaxTabuSize;

	return CFixedTimeSearch::Solve(InitialSolution, 
							   dTimeoutInSeconds, 
							   ulMaxIterations,
							   ulSeed);
}

template<typename SolutionType, typename ValueType> 
CTabuSearch<SolutionType, ValueType>::CTabuSearch(const string strPostFileName)
	:CFixedTimeSearch(strPostFileName)
{
	m_ulNeighborsCount = 0;
	m_ulTabuSize = 0;
	m_ulNextPushIndex = 0;
}

template<typename SolutionType, typename ValueType> 
void CTabuSearch<SolutionType, ValueType>::TabuPushAttribute(float fAttribute)
{
	m_fTabuList[m_ulNextPushIndex] = fAttribute;
	m_ulNextPushIndex = (m_ulNextPushIndex + 1) % m_ulMaxTabuSize;
	m_ulTabuSize = min(m_ulTabuSize+1,m_ulMaxTabuSize);
}

template<typename SolutionType, typename ValueType> 
bool CTabuSearch<SolutionType, ValueType>::TabuContainsAttribute(float fAttribute)
{
	for (unsigned long ulTabuItem = 0; ulTabuItem < m_ulTabuSize; ulTabuItem++)
	{
		if (abs(m_fTabuList[ulTabuItem]-fAttribute)<=0.5)
			return true;
	}

	return false;
}

template<typename SolutionType, typename ValueType> 
void CTabuSearch<SolutionType, ValueType>::ChooseCandidate(SolutionType& Chosen)
{
	//Generate up to MAX_NEIGHBORS_COUNT neighbors to m_CurrentNeighbors, and the actual number of neighbors created to m_ulNeighborsCount
	GenerateNeighbors();

	SolutionType BestCandidate;
	ValueType BestCandidateValue, CurrentCandidateValue;
	BestCandidate = m_CurrentNeighbors[0];
	CountedEvaluateSolution(BestCandidate, BestCandidateValue);

	for (unsigned long ulCandidate=0; ulCandidate < m_ulNeighborsCount; ulCandidate++)
	{
		CountedEvaluateSolution(m_CurrentNeighbors[ulCandidate], CurrentCandidateValue);
		if ((CurrentCandidateValue < BestCandidateValue) && (!TabuForbidsSolution(m_CurrentNeighbors[ulCandidate]))
			|| (CurrentCandidateValue < m_ValueOfBestSolution))
		{
 			BestCandidate = m_CurrentNeighbors[ulCandidate];
			BestCandidateValue = CurrentCandidateValue;
		}
		TabuUpdate(m_CurrentNeighbors[ulCandidate]);
	}
	
	Chosen = BestCandidate;
	return;

}

template<typename SolutionType, typename ValueType> 
bool CTabuSearch<SolutionType, ValueType>::AcceptanceCritirionReached(const SolutionType& Solution) 
{
	ValueType ValueOfSolution;
	CountedEvaluateSolution(Solution, ValueOfSolution);

	//Minimization Problem
	if ((ValueOfSolution < m_ValueOfBestSolution) || (!TabuForbidsSolution(Solution)))
	{
		TabuUpdate(Solution);
		return true;
	}

	return false;
}


#pragma endregion
#pragma endregion

#pragma region COP Local search
#pragma region CCOPRandomSearch
///
/// A CCOPRandomSearch class - A random search for Constrained Optimization Problems , inherits from CFixedTimeSearch
/// This concrete class:
/// 1. Define the two template typenames : SolutionType as int* and ValueType as float*
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///

class CCOPRandomSearch : public CFixedTimeSearch <COP::SolutionVector, COP::GradesVector> 
{

public:
	
	CCOPRandomSearch(const string strPostFileName);

	virtual double Solve(const COP::SolutionVector& InitialSolution, 
					     COP& Cop,
					     double dTimeoutInSeconds, 
					     unsigned long ulMaxIterations,
					     unsigned long ulSeed
					     );

	virtual void ChooseCandidate(COP::SolutionVector& Chosen);
	virtual void EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual bool AcceptanceCritirionReached(const COP::SolutionVector& Solution);
	virtual void Restart(COP::SolutionVector& NewInitial); 

	//Problem input
	COP* m_Cop;
};

CCOPRandomSearch::CCOPRandomSearch(const string strPostFileName)
	: CFixedTimeSearch(strPostFileName)
{	
}

double CCOPRandomSearch::Solve(const COP::SolutionVector& InitialSolution, 
							   COP& Cop,
							   double dTimeoutInSeconds, 
							   unsigned long ulMaxIterations,
							   unsigned long ulSeed
							  )
{

	//*m_Cop = Cop;
	m_Cop = &Cop;

	///Problem Instance end

	return CFixedTimeSearch::Solve(InitialSolution,
							   dTimeoutInSeconds, 
							   ulMaxIterations,
							   ulSeed
							   );
}

//Choose a the best neighbor in the partial solution, considering all values of the current variable
void CCOPRandomSearch::ChooseCandidate(COP::SolutionVector& Chosen)
{
	//A random solution is chosen
	for (unsigned short usCurrentVar = 0; usCurrentVar<COP::MAX_NUM_OF_VARS; usCurrentVar++)
	{
		std::uniform_int_distribution<uint32_t> UIntDistForValue(0,m_Cop->ValuesPerVariables->VarsData[usCurrentVar].ulValuesAmount);
		unsigned long ulRandomValue = UIntDistForValue(AlgorithmRandomEngine);
		Chosen.aulSolutionVector[usCurrentVar] = ulRandomValue;
	}
}

void CCOPRandomSearch::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation)
{
	m_Cop->EvaluateSolution(Solution,Evaluation);
	return;

}

bool CCOPRandomSearch::StopCritirionReached()
{
	return (m_ulIterations >= m_ulMaxIterations); //|| (m_fTemprature < 0.1) if temprature might be negative
}

bool CCOPRandomSearch::RestartCritirionReached()
{
	//Always restart
	return true;
}

bool CCOPRandomSearch::AcceptanceCritirionReached(const COP::SolutionVector& Solution)
{
	//Always accept solution
	return true;
}

void CCOPRandomSearch::Restart(COP::SolutionVector& NewInitial)
{
	return;
}
#pragma endregion

#pragma region CCOPRandomWalk
///
/// A CCOPRandomWalk class - A random walk for Constrained Optimization Problems , inherits from CFixedTimeSearch
/// This concrete class:
/// 1. Define the two template typenames : SolutionType as int* and ValueType as float*
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///

class CCOPRandomWalk : public CFixedTimeSearch <COP::SolutionVector, COP::GradesVector> 
{

public:

	CCOPRandomWalk(const string strPostFileName);

	virtual double Solve(const COP::SolutionVector& InitialSolution, 
					     COP& Cop,
					     double dTimeoutInSeconds, 
					     unsigned long ulMaxIterations,
					     unsigned int uiNumOfVarChangesInNeighborhood,
					     unsigned long ulSeed
					     );

	virtual void ChooseCandidate(COP::SolutionVector& Chosen);
	virtual void EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual bool AcceptanceCritirionReached(const COP::SolutionVector& Solution);
	virtual void Restart(COP::SolutionVector& NewInitial); 

	unsigned int m_uiNumOfVarChangesInNeighborhood;

	//Problem input
	COP* m_Cop;
};

CCOPRandomWalk::CCOPRandomWalk(const string strPostFileName)
	: CFixedTimeSearch(strPostFileName)
{	
}

double CCOPRandomWalk::Solve(const COP::SolutionVector& InitialSolution, 
						     COP& Cop,
						     double dTimeoutInSeconds, 
						     unsigned long ulMaxIterations,
						     unsigned int uiNumOfVarChangesInNeighborhood,
						     unsigned long ulSeed
						    )
{

	//*m_Cop = Cop;
	m_Cop = &Cop;

	///Problem Instance end

	m_uiNumOfVarChangesInNeighborhood = uiNumOfVarChangesInNeighborhood;
	return CFixedTimeSearch::Solve(InitialSolution,
							   dTimeoutInSeconds, 
							   ulMaxIterations,
							   ulSeed
							   );
}

//Choose a the best neighbor in the partial solution, considering all values of the current variable
void CCOPRandomWalk::ChooseCandidate(COP::SolutionVector& Chosen)
{
	//A random negibor is chosen
	m_Cop->GenerateSingleNeighbor(Chosen, m_CurrentSolution, m_uiNumOfVarChangesInNeighborhood, AlgorithmRandomEngine);
	return;
}

void CCOPRandomWalk::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation)
{
	m_Cop->EvaluateSolution(Solution,Evaluation);
	return;

}

bool CCOPRandomWalk::StopCritirionReached()
{
	return (m_ulIterations >= m_ulMaxIterations); //|| (m_fTemprature < 0.1) if temprature might be negative
}

bool CCOPRandomWalk::RestartCritirionReached()
{
	//Always restart
	return true;
}

bool CCOPRandomWalk::AcceptanceCritirionReached(const COP::SolutionVector& Solution)
{
	//Always accept solution
	return true;
}

void CCOPRandomWalk::Restart(COP::SolutionVector& NewInitial)
{
	return;
}
#pragma endregion

#pragma region CCOPGreedySearch
///
/// A CCOPGreedySearch class - A greedy search for Constrained Optimization Problems , inherits from CFixedTimeSearch
/// This concrete class:
/// 1. Define the two template typenames : SolutionType as int* and ValueType as float*
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///

class CCOPGreedySearch : public CFixedTimeSearch <COP::SolutionVector, COP::GradesVector> 
{

public:

	CCOPGreedySearch(const string strFileNamePost);

	virtual double Solve(const COP::SolutionVector& InitialSolution, 
					     COP& Cop,
					     double dTimeoutInSeconds, 
					     unsigned long ulMaxIterations,
					     unsigned long ulSeed,
						 bool bLoopy
					    );

	virtual void ChooseCandidate(COP::SolutionVector& Chosen);
	virtual void EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual bool AcceptanceCritirionReached(const COP::SolutionVector& Solution);
	virtual void Restart(COP::SolutionVector& NewInitial); 
	unsigned short m_usCurrentVariable;
	bool m_bLoopy;

	//Problem input
	COP* m_Cop;
};

CCOPGreedySearch::CCOPGreedySearch(const string strPostFileName)
	: CFixedTimeSearch(strPostFileName)
{
}

double CCOPGreedySearch::Solve(const COP::SolutionVector& InitialSolution, 
							   COP& Cop,
							   double dTimeoutInSeconds, 
							   unsigned long ulMaxIterations,
							   unsigned long ulSeed,
							   bool bLoopy
							  )
{

	//*m_Cop = Cop;
	m_Cop = &Cop;
	m_bLoopy = bLoopy;

	//Sort by priorities - Since the greedy algorithm works by variable order  // (arr, num_elements, sizeof_elem, cmp_func)
	qsort(m_Cop->ValuesPerVariables->VarsData,m_Cop->ValuesPerVariables->uiValidVarAmount,sizeof(COP::VarData),CompareVarByPrio);

	m_usCurrentVariable = 0;
	double dResult = CFixedTimeSearch::Solve(InitialSolution,
							   dTimeoutInSeconds, 
							   ulMaxIterations,
							   ulSeed
							   );

	return dResult;
}

//Choose a the best neighbor in the partial solution, considering all values of the current variable
void CCOPGreedySearch::ChooseCandidate(COP::SolutionVector& Chosen)
{
	COP::SolutionVector sNeighbor = m_CurrentSolution;
	COP::GradesVector sNeighborValue = m_ValueOfCurrentSolution;
	COP::SolutionVector sBestNeighborSolution;
	m_Cop->WorstSolution(sBestNeighborSolution);
	COP::GradesVector sBestNeighborValue; 
	m_Cop->WorstValue(sBestNeighborValue);
	
	for (unsigned short usCurrentVarValue=0; usCurrentVarValue<m_Cop->uiMaxValuesNum; usCurrentVarValue++)
	{
		sNeighbor.aulSolutionVector[m_usCurrentVariable] = usCurrentVarValue;
		CountedEvaluateSolution(sNeighbor, sNeighborValue);
		if (sNeighborValue < sBestNeighborValue)
		{
			sBestNeighborSolution = sNeighbor;
			sBestNeighborValue = sNeighborValue;
		}
	}

	m_usCurrentVariable++;
	
	Chosen = sBestNeighborSolution;
	return;
}

void CCOPGreedySearch::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation)
{
	m_Cop->EvaluateSolution(Solution,Evaluation);
	return;

}

bool CCOPGreedySearch::StopCritirionReached()
{
	if (m_ulIterations >= m_ulMaxIterations) return true;
	if (m_bLoopy) return false;
	return (m_usCurrentVariable >= m_Cop->ValuesPerVariables->uiValidVarAmount); 
}

bool CCOPGreedySearch::RestartCritirionReached()
{
	if (m_bLoopy) return (m_usCurrentVariable >= m_Cop->ValuesPerVariables->uiValidVarAmount);
	return false;
}

bool CCOPGreedySearch::AcceptanceCritirionReached(const COP::SolutionVector& Solution)
{
	COP::GradesVector ValueOfSolution; 
	CountedEvaluateSolution(Solution, ValueOfSolution);

	return (ValueOfSolution < m_ValueOfBestSolution) || m_bLoopy;
}

void CCOPGreedySearch::Restart(COP::SolutionVector& NewInitial)
{
	std::fill(NewInitial.aulSolutionVector,NewInitial.aulSolutionVector+COP::MAX_NUM_OF_VARS,0);
	//Sort by priorities - Since the greedy algorithm works by variable order
	qsort(m_Cop->ValuesPerVariables->VarsData,m_Cop->ValuesPerVariables->uiValidVarAmount,sizeof(COP::VarData),CompareVarByPrio);
	m_usCurrentVariable = 0;
	return;
}
#pragma endregion

#pragma region CCOPSimulatedAnnealingSearch
///
/// A CCOPSimulatedAnnealingSearch class - A simulated annealing search for Constrained Optimization Problems , inherits from CSimulatedAnnealingSearch
/// This concrete class:
/// 1. Define the two template typenames : SolutionType as int* and ValueType as float*
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///

class CCOPSimulatedAnnealingSearch : public CSimulatedAnnealingSearch <COP::SolutionVector, COP::GradesVector> 
{

public:

	CCOPSimulatedAnnealingSearch(const string strPostFileName);

	virtual double Solve(const COP::SolutionVector& InitialSolution, 
					     COP& Cop,
					     double dTimeoutInSeconds, 
					     unsigned long ulMaxIterations,
					     float fInitialTemprature, 
					     float fTempratureStep,
					     unsigned int uiNumOfVarChangesInNeighborhood,
					     unsigned long ulSeed);

	virtual void ChooseCandidate(COP::SolutionVector& Chosen);
	virtual void EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual void Restart(COP::SolutionVector& NewInitial); 

	unsigned int m_uiNumOfVarChangesInNeighborhood;

	//Problem input
	COP* m_Cop;
};

CCOPSimulatedAnnealingSearch::CCOPSimulatedAnnealingSearch(const string strPostFileName)
	: CSimulatedAnnealingSearch(strPostFileName)
{
	
}


double CCOPSimulatedAnnealingSearch::Solve(const COP::SolutionVector& InitialSolution, 
										   COP& Cop,
										   double dTimeoutInSeconds, 
										   unsigned long ulMaxIterations,
										   float fInitialTemprature, 
										   float fTempratureStep,
										   unsigned int uiNumOfVarChangesInNeighborhood,
										   unsigned long ulSeed
										   )
{

	//*m_Cop = Cop;
	m_Cop = &Cop;

	///Problem Instance end
	
	m_uiNumOfVarChangesInNeighborhood = uiNumOfVarChangesInNeighborhood;
	return CSimulatedAnnealingSearch::Solve(InitialSolution,
											dTimeoutInSeconds, 
											ulMaxIterations,
											fInitialTemprature, 
											fTempratureStep,
											ulSeed);
}

void CCOPSimulatedAnnealingSearch::ChooseCandidate(COP::SolutionVector& Chosen)
{
	//Choose a neighbor uniformaly at random with m_uiNumOfValChangesInNeighborhood variable changes s	
	m_Cop->GenerateSingleNeighbor(Chosen, m_CurrentSolution, m_uiNumOfVarChangesInNeighborhood,AlgorithmRandomEngine);

	return;
}

void CCOPSimulatedAnnealingSearch::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation)
{

	m_Cop->EvaluateSolution(Solution, Evaluation);
	return;

}

bool CCOPSimulatedAnnealingSearch::StopCritirionReached()
{
	return (m_ulIterations >= m_ulMaxIterations); //|| (m_fTemprature < 0.1) if temprature might be negative
}

bool CCOPSimulatedAnnealingSearch::RestartCritirionReached()
{
	//No restarts
	return false;
}

void CCOPSimulatedAnnealingSearch::Restart(COP::SolutionVector& NewInitial)
{

	std::fill(NewInitial.aulSolutionVector,NewInitial.aulSolutionVector+COP::MAX_NUM_OF_VARS,0);
	return;
}
#pragma endregion

#pragma region CCOPTabuSearch
///
/// A CCOPTabuSearch class - A simulated annealing search for Constrained Optimization Problems , inherits from CSimulatedAnnealingSearch
/// This concrete class:
/// 1. Define the two template typenames : SolutionType as int* and ValueType as float*
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///

class CCOPTabuSearch : public CTabuSearch <COP::SolutionVector, COP::GradesVector> 
{

public:

	CCOPTabuSearch(const string strPostFileName);

	virtual double Solve(const COP::SolutionVector& InitialSolution, 
					     COP& Cop, 
					     unsigned int uiNumOfVarChangesInNeighborhood,
					     double dTimeoutInSeconds,
					     unsigned long ulMaxIterations,
					     unsigned long ulMaxTabuSize,
					     unsigned long ulSeed);

	virtual void ChooseCandidate(COP::SolutionVector& Chosen);
	virtual void GenerateNeighbors();
	virtual bool TabuForbidsSolution(const COP::SolutionVector& Solution); // should use TabuContainsAttributes
	virtual void TabuUpdate(const COP::SolutionVector& Solution); //should use Tabu PushAttribute
	virtual void EvaluateSolution(const COP::SolutionVector& Solution,COP::GradesVector& Evaluation);
	virtual bool AcceptanceCritirionReached(const COP::SolutionVector& Solution);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();

	virtual void Restart(COP::SolutionVector& NewInitial);

	unsigned int m_uiNumOfVarChangesInNeighborhood;

	//Problem input
	COP* m_Cop;
};

CCOPTabuSearch::CCOPTabuSearch(const string strPostFileName)
	: CTabuSearch(strPostFileName)
{
	
}

double CCOPTabuSearch::Solve(const COP::SolutionVector& InitialSolution, 
						     COP& Cop, 
						     unsigned int uiNumOfVarChangesInNeighborhood,
						     double dTimeoutInSeconds,
						     unsigned long ulMaxIterations,
						     unsigned long ulMaxTabuSize,
						     unsigned long ulSeed)
{

	//*m_Cop = Cop;
	m_Cop = &Cop;

	m_uiNumOfVarChangesInNeighborhood = uiNumOfVarChangesInNeighborhood;

	return CTabuSearch::Solve(InitialSolution,
							  dTimeoutInSeconds,
							  ulMaxIterations,
							  ulMaxTabuSize,
							  ulSeed);
}

void CCOPTabuSearch::ChooseCandidate(COP::SolutionVector& Chosen)
{
	//Choose a neighbor uniformaly at random with m_uiNumOfValChangesInNeighborhood variable changes 	
	m_Cop->GenerateSingleNeighbor(Chosen, m_CurrentSolution, m_uiNumOfVarChangesInNeighborhood,AlgorithmRandomEngine);
	return;

	/*while (true)
	{
		m_Cop->GenerateSingleNeighbor(sNeighbor, m_CurrentSolution, m_uiNumOfVarChangesInNeighborhood);
	
		COP::GradesVector ValueOfNeighbor = EvaluateSolution(sNeighbor);

		if ((ValueOfNeighbor < m_ValueOfBestSolution) || (!TabuForbidsSolution(sNeighbor)))
		{
			TabuUpdate(sNeighbor);
			return sNeighbor;
		}
	}*/
}

bool CCOPTabuSearch::AcceptanceCritirionReached(const COP::SolutionVector& Solution)
{
	COP::GradesVector ValueOfSolution;
	CountedEvaluateSolution(Solution,ValueOfSolution);

	if ((ValueOfSolution < m_ValueOfBestSolution) || (!TabuForbidsSolution(Solution)))
	{
		TabuUpdate(Solution);
		return true;
	}

	return false;

	//return (ValueOfSolution < m_ValueOfBestSolution);
}

void CCOPTabuSearch::GenerateNeighbors()
{
	m_ulNeighborsCount = 0;
	COP::SolutionVector sNeighbor = {0};
	for (unsigned int uiCurrentNeighbor=0; uiCurrentNeighbor<MAX_NEIGHBORS_COUNT; uiCurrentNeighbor++)
	{
		//Choose a neighbor uniformaly at random with m_uiNumOfValChangesInNeighborhood variable changes 
		m_Cop->GenerateSingleNeighbor(sNeighbor,m_CurrentSolution,m_uiNumOfVarChangesInNeighborhood,AlgorithmRandomEngine);
		
		m_CurrentNeighbors[uiCurrentNeighbor]  = sNeighbor;
		m_ulNeighborsCount++;
	}
}

bool CCOPTabuSearch::TabuForbidsSolution(const COP::SolutionVector& Solution)
{
	COP::SolutionVector Delta;
	Delta = Solution - m_CurrentSolution;
	for (unsigned int uiIndex=0; uiIndex < COP::MAX_NUM_OF_VARS; uiIndex++)
	{
		if ((Delta.aulSolutionVector[uiIndex] > FLT_EPSILON) && TabuContainsAttribute((float)uiIndex))
			return true;
	}
	
	return false;
}

void CCOPTabuSearch::TabuUpdate(const COP::SolutionVector& Solution)
{
	COP::SolutionVector Delta;
	Delta = Solution - m_CurrentSolution;
	for (unsigned int uiIndex=0; uiIndex < COP::MAX_NUM_OF_VARS; uiIndex++)
	{
		if (Delta.aulSolutionVector[uiIndex] > FLT_EPSILON)
			TabuPushAttribute((float)uiIndex);
	}
	
}

void CCOPTabuSearch::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation)
{
	m_Cop->EvaluateSolution(Solution, Evaluation);
	return;
}

bool CCOPTabuSearch::StopCritirionReached()
{
	return (m_ulIterations >= m_ulMaxIterations); //|| (m_fTemprature < 0.1) if temprature might be negative
}

bool CCOPTabuSearch::RestartCritirionReached()
{
	//No restarts
	return false;
}

void CCOPTabuSearch::Restart(COP::SolutionVector& NewInitial)
{
	std::fill(NewInitial.aulSolutionVector,NewInitial.aulSolutionVector+COP::MAX_NUM_OF_VARS,0);
	return;
}
#pragma endregion

#pragma region CCOPStochasticHillClimbing
///
/// A CCOPStochastiCFixedTimeSearch class - A simulated annealing search for Constrained Optimization Problems , inherits from CSimulatedAnnealingSearch
/// This concrete class:
/// 1. Define the two template typenames : SolutionType as int* and ValueType as float*
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///

class CCOPStochasticHillClimbing : public CFixedTimeSearch <COP::SolutionVector, COP::GradesVector> 
{

public:
	
	CCOPStochasticHillClimbing(const string strPostFileName);
	~CCOPStochasticHillClimbing();

	virtual double Solve(const COP::SolutionVector& InitialSolution, 
						 COP& Cop,
						 double dTimeoutInSeconds, 
						 unsigned long ulMaxIterations,
						 unsigned int uiNumOfVarChangesInNeighborhood,
					     unsigned long ulSeed);

	virtual void ChooseCandidate(COP::SolutionVector& Chosen);
	virtual void EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation);
	virtual bool AcceptanceCritirionReached(const COP::SolutionVector& Solution);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual void Restart(COP::SolutionVector& NewwInitial);

	unsigned int m_uiNumOfVarChangesInNeighborhood;

	//Problem input
	COP* m_Cop;
};

CCOPStochasticHillClimbing::CCOPStochasticHillClimbing(const string strPostFileName)
	: CFixedTimeSearch(strPostFileName)
{
	
}

CCOPStochasticHillClimbing::~CCOPStochasticHillClimbing()
{
	
}
//HELLO
double CCOPStochasticHillClimbing::Solve(const COP::SolutionVector& InitialSolution, 
										 COP& Cop,
										 double dTimeoutInSeconds, 
										 unsigned long ulMaxIterations,
										 unsigned int uiNumOfVarChangesInNeighborhood,
									     unsigned long ulSeed)
{
	//*m_Cop = Cop;
	m_Cop = &Cop;
	
	m_uiNumOfVarChangesInNeighborhood = uiNumOfVarChangesInNeighborhood;
	return CFixedTimeSearch::Solve(InitialSolution,
							  dTimeoutInSeconds,
							  ulMaxIterations,
							  ulSeed);
}

void CCOPStochasticHillClimbing::ChooseCandidate(COP::SolutionVector& Chosen)
{
	//Choose a neighbor uniformaly at random with m_uiNumOfValChangesInNeighborhood variable changes 

	m_Cop->GenerateSingleNeighbor(Chosen, m_CurrentSolution, m_uiNumOfVarChangesInNeighborhood,AlgorithmRandomEngine);
	COP::GradesVector ValueOfNeighbor; 
	CountedEvaluateSolution(Chosen, ValueOfNeighbor);

	return;
}

void CCOPStochasticHillClimbing::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation)
{
	m_Cop->EvaluateSolution(Solution, Evaluation);
	return;
}

bool CCOPStochasticHillClimbing::AcceptanceCritirionReached(const COP::SolutionVector& Solution)
{
	COP::GradesVector SolutionValue; 
	CountedEvaluateSolution(Solution,SolutionValue); 
	return (SolutionValue < m_ValueOfCurrentSolution);
}

bool CCOPStochasticHillClimbing::StopCritirionReached()
{
	return (m_ulIterations >= m_ulMaxIterations); 
}

bool CCOPStochasticHillClimbing::RestartCritirionReached()
{
	//No restarts
	return false;
}

void CCOPStochasticHillClimbing::Restart(COP::SolutionVector& NewInitial)
{
	std::fill(NewInitial.aulSolutionVector,NewInitial.aulSolutionVector+COP::MAX_NUM_OF_VARS,0);
	return;
}
#pragma endregion

#pragma region CCOPCrossEntropy
///
/// A CCOPCrossEntropy class - A simulated annealing search for Constrained Optimization Problems , inherits from CSimulatedAnnealingSearch
/// This concrete class:
/// 1. Define the two template typenames : SolutionType as int* and ValueType as float*
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///

typedef struct Sample
{
	COP::SolutionVector Solution;
	COP::GradesVector Grade;
} Sample;

class CCOPCrossEntropy : public CFixedTimeSearch <COP::SolutionVector, COP::GradesVector> 
{

public:
	
	static const unsigned int MAX_SAMPLES_NUM = 200;

	CCOPCrossEntropy(const string strPostFileName);
	~CCOPCrossEntropy();

	virtual double Solve(const COP::SolutionVector& InitialSolution,
					     COP& Cop,
					     double dTimeoutInSeconds, 
					     unsigned long ulMaxIterations,
					     unsigned int uiSamples,
					     float fInitialSolutionsWeight,
					     float fAlpha,
					     float fRho,
					     float fEpsilon,
					     unsigned long ulSeed);

	virtual void ChooseCandidate(COP::SolutionVector& Chosen);
	virtual void EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation);
	virtual bool AcceptanceCritirionReached(const COP::SolutionVector& Solution);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual void Restart(COP::SolutionVector& NewInitial);

	virtual void InitDistributionFunctions(const COP::SolutionVector& InitialSolution,float fInitialSolutionsWeight );
	virtual void CalcCDF();
	virtual bool UpdateProbabilities();
	virtual unsigned int GetVarValue(double dUniformRnd, 
									 unsigned int uiMin, 
									 unsigned int uiMax, 
									 const float* aVarCdf);
	virtual void GenerateSolutions(bool bUseInitialSolution);
	virtual void GenerateSolutionFromInitial(const COP::SolutionVector& Initial);
	virtual void SortSamples();

	//Problem input
	COP* m_Cop;

	unsigned int m_uiSamples;
	float m_fInitialSolutionWeight;
	float m_fAlpha;
	float m_fRho;
	float m_fEpsilon;
	

	float m_aafPdf[COP::MAX_NUM_OF_VARS][COP::MAX_VALUES_OF_VAR]; // f_X(x) = P(X = x)
	float m_aafPdfElite[COP::MAX_NUM_OF_VARS][COP::MAX_VALUES_OF_VAR]; // f_X(x) = P(X = x)
	float m_aafCdf[COP::MAX_NUM_OF_VARS][COP::MAX_VALUES_OF_VAR]; // F_X(x) = P(X <= x)
	Sample m_aSamples[MAX_SAMPLES_NUM]; // samples from that Pdf
	bool m_bConverged;

};

CCOPCrossEntropy::CCOPCrossEntropy(const string strPostFileName)
	: CFixedTimeSearch(strPostFileName)
{
	
}

CCOPCrossEntropy::~CCOPCrossEntropy()
{
	
}

void CCOPCrossEntropy::InitDistributionFunctions(const COP::SolutionVector& InitialSolution,float fInitialSolutionsWeight ) 
{
	//Elite PDF init
	memset(&m_aafPdfElite,0, sizeof(float)*COP::MAX_NUM_OF_VARS*COP::MAX_VALUES_OF_VAR);

	if (fInitialSolutionsWeight <= FLT_EPSILON) {
		//PDF init without Initial Solution
		for (unsigned int uiVariable=0; uiVariable<m_Cop->ValuesPerVariables->uiValidVarAmount; uiVariable++)		
			for (unsigned int uiValue=0; uiValue<m_Cop->ValuesPerVariables->VarsData[uiVariable].ulValuesAmount; uiValue++)
			{
				m_aafPdf[uiVariable][uiValue] = 1/(float)m_Cop->ValuesPerVariables->VarsData[uiVariable].ulValuesAmount;
				m_aafCdf[uiVariable][uiValue] = 0;
			}
	}
	else //fInitialSolutionsWeight > FLT_EPSILON
	{
		//PDF init with Initial Solution
		for (unsigned int uiVariable=0; uiVariable<m_Cop->ValuesPerVariables->uiValidVarAmount; uiVariable++)		
			for (unsigned int uiValue=0; uiValue<m_Cop->ValuesPerVariables->VarsData[uiVariable].ulValuesAmount; uiValue++)
			{
				m_aafPdf[uiVariable][uiValue] = (1-fInitialSolutionsWeight)/(float)m_Cop->ValuesPerVariables->VarsData[uiVariable].ulValuesAmount;
				if (InitialSolution.aulSolutionVector[uiVariable] == uiValue)
					m_aafPdf[uiVariable][uiValue] += fInitialSolutionsWeight;

				m_aafCdf[uiVariable][uiValue] = 0;
			}
	}

	CalcCDF();
}

void CCOPCrossEntropy::CalcCDF()
{
  for (unsigned int uiVariable = 0; uiVariable < m_Cop->ValuesPerVariables->uiValidVarAmount; uiVariable++)
    {
      float fTempSum = 0;
      for(unsigned int uiValue=0; uiValue<m_Cop->ValuesPerVariables->VarsData[uiVariable].ulValuesAmount; uiValue++)
        {			
			m_aafCdf[uiVariable][uiValue] = fTempSum + m_aafPdf[uiVariable][uiValue];
			fTempSum += m_aafPdf[uiVariable][uiValue];          
        }	

      // for numerical stability
	  m_aafCdf[uiVariable][m_Cop->ValuesPerVariables->VarsData[uiVariable].ulValuesAmount-1] = 1.0;
    }
}

bool CCOPCrossEntropy::UpdateProbabilities()
{
  bool bStop = true;

  // set temporary vector to zero
  for (unsigned int uiVariable = 0; uiVariable < m_Cop->ValuesPerVariables->uiValidVarAmount; uiVariable++)
    {
		 for(unsigned int uiValue=0; uiValue<m_Cop->ValuesPerVariables->VarsData[uiVariable].ulValuesAmount; uiValue++)
        {
			m_aafPdfElite[uiVariable][uiValue] = 0;
        }
    }

  // builds a new distribution space only over the variables in the
  // elite set (the rest are with probability 0). Note that later it
  // is blended with the old distribution via smoothing.
  int iStartEliteIndex = (int)floor((1-m_fRho)*m_uiSamples);
  float fEliteProbDelta = 1/((float)m_uiSamples-(float)iStartEliteIndex);
 
  for (unsigned int uiEliteIndex = iStartEliteIndex; uiEliteIndex<m_uiSamples; uiEliteIndex++) // going over the elite set
    {
		  for (unsigned int uiVariable = 0; uiVariable<m_Cop->ValuesPerVariables->uiValidVarAmount; uiVariable++)  // going over the variables
        {
			m_aafPdfElite[uiVariable][m_aSamples[uiEliteIndex].Solution.aulSolutionVector[uiVariable]] += fEliteProbDelta;
        }
    }

  // blending with the old distribution via smoothing. 
	for (unsigned int uiVariable = 0; uiVariable < m_Cop->ValuesPerVariables->uiValidVarAmount; uiVariable++) // for each variable
    {
      double dMaxVal = 0;
	  // stop becomes false if at least for one variable the prob. is smaller than 1 - m_eps.
      for (unsigned int uiValue=0; uiValue < m_Cop->ValuesPerVariables->VarsData[uiVariable].ulValuesAmount; uiValue++) // for each domain value
        {
			m_aafPdf[uiVariable][uiValue] = m_fAlpha * m_aafPdfElite[uiVariable][uiValue] + (1 - m_fAlpha) * m_aafPdf[uiVariable][uiValue];			
			if(bStop && m_aafPdf[uiVariable][uiValue] > dMaxVal) 
				dMaxVal = m_aafPdf[uiVariable][uiValue]; // if stop=false becase of a previous variable, no point keep checking. 
        }
	  
      if (bStop && dMaxVal < 1.0 - m_fEpsilon) 
		 bStop = false;	
    }	
  
	if (!bStop) 
	  CalcCDF();

  return bStop;
}

//Binary / Weighted Search
unsigned int CCOPCrossEntropy::GetVarValue(double dUniformRnd, 
										   unsigned int uiMin, 
										   unsigned int uiMax, 
										   const float* aVarCdf)
{
	double dRatio;
	unsigned int uiMid;
	if (uiMin == uiMax || dUniformRnd <= aVarCdf[uiMin]) return uiMin;
	dRatio = (dUniformRnd - aVarCdf[uiMin]) / (aVarCdf[uiMax] - aVarCdf[uiMin]);	
	uiMid = (uiMin + (int)((uiMax - uiMin) * dRatio));
	if (dUniformRnd < aVarCdf[uiMid]) {
		if (dUniformRnd >= aVarCdf[uiMid - 1]) return uiMid;
		else return GetVarValue(dUniformRnd, uiMin, (uiMid - 1), aVarCdf);
	}
	if (dUniformRnd > aVarCdf[uiMid]) {
		if (dUniformRnd <= aVarCdf[uiMid + 1]) return (uiMid + 1);
		else return GetVarValue(dUniformRnd, (uiMid + 1), uiMax, aVarCdf);
	}
	else return uiMid;
}

void CCOPCrossEntropy::GenerateSolutionFromInitial(const COP::SolutionVector& Initial)
{
	m_aSamples[0].Solution = Initial;
	CountedEvaluateSolution(m_aSamples[0].Solution,m_aSamples[0].Grade);
}

void CCOPCrossEntropy::GenerateSolutions(bool bUseInitialSolution)
{

  unsigned int uiSample;
  unsigned int uiIndex;
  unsigned int uiPrevIndex = 0;
		
  // first time we call this function, we want it to take into account
  // the hint as the first trajectory. It is put in there through
  // GenerateTrajectoryFromHints
  unsigned int uiStartIdx = bUseInitialSolution ? 1 : 0;  

  for(unsigned int uiVariable = 0; uiVariable < m_Cop->ValuesPerVariables->uiValidVarAmount; uiVariable++)		
    {				
      
		for (uiSample = uiStartIdx; uiSample < m_uiSamples; uiSample++) 
        {
		  std::uniform_real_distribution<float> fDistribution(0.0,1.0);
          float fUniformRnd =  fDistribution(AlgorithmRandomEngine); // uniform random value in the range [0,1]

          // if the chosen value happens to be the same index as the
          // previous one, it saves us the search in
          // GetInd. Statistically it is more than 30% of the cases.
		  if (uiPrevIndex > 0 && fUniformRnd >= m_aafCdf[uiVariable][uiPrevIndex-1] && fUniformRnd <= m_aafCdf[uiVariable][uiPrevIndex])
			  uiIndex = uiPrevIndex; 
          else        
			  uiIndex = GetVarValue(fUniformRnd, 0, (m_Cop->ValuesPerVariables->VarsData[uiVariable].ulValuesAmount-1), m_aafCdf[uiVariable]);

          uiPrevIndex = uiIndex;	
		  m_aSamples[uiSample].Solution.aulSolutionVector[uiVariable] = uiIndex;
        }		
    }
  
  for (unsigned int uiSample = 0; uiSample < m_uiSamples; uiSample++) 
  {
	  CountedEvaluateSolution(m_aSamples[uiSample].Solution,m_aSamples[uiSample].Grade);
  }
 
}

inline int CompareSamples (const void * SampleA, const void * SampleB)
{
  Sample* pSampleA = (Sample *)SampleA;
  Sample *pSampleB = (Sample *)SampleB;

  if (pSampleA->Grade > pSampleB->Grade) return -1;
  if (pSampleA->Grade < pSampleB->Grade) return 1;
  return 0;
}

void CCOPCrossEntropy::SortSamples()
{
	qsort(m_aSamples,m_uiSamples,sizeof(Sample),CompareSamples);
}

double CCOPCrossEntropy::Solve(const COP::SolutionVector& InitialSolution,
							   COP& Cop,
							   double dTimeoutInSeconds, 
							   unsigned long ulMaxIterations,
							   unsigned int uiSamples,
							   float fInitialSolutionsWeight,
							   float fAlpha,
							   float fRho,
							   float fEpsilon,
							   unsigned long ulSeed)
{
	//*m_Cop = Cop;
	m_Cop = &Cop;
	
	assert(uiSamples <= MAX_SAMPLES_NUM);

	m_uiSamples = uiSamples;
	m_fInitialSolutionWeight = fInitialSolutionsWeight;
	m_fAlpha = fAlpha;
	m_fRho = fRho;
	m_fEpsilon = fEpsilon;

	InitDistributionFunctions(InitialSolution,fInitialSolutionsWeight);
	GenerateSolutionFromInitial(InitialSolution);

	return CFixedTimeSearch::Solve(InitialSolution,
							  dTimeoutInSeconds,
							  ulMaxIterations,
							  ulSeed);
}

void CCOPCrossEntropy::ChooseCandidate(COP::SolutionVector& Chosen)
{
	GenerateSolutions(true);
	SortSamples();
	m_bConverged = UpdateProbabilities();
	Chosen = m_aSamples[m_uiSamples-1].Solution;
}

void CCOPCrossEntropy::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation)
{
	m_Cop->EvaluateSolution(Solution, Evaluation);
	return;
}

bool CCOPCrossEntropy::AcceptanceCritirionReached(const COP::SolutionVector& Solution)
{
	COP::GradesVector SolutionValue; 
	CountedEvaluateSolution(Solution,SolutionValue); 
	return (SolutionValue < m_ValueOfCurrentSolution);
}

bool CCOPCrossEntropy::StopCritirionReached()
{
	return (m_bConverged || (m_ulIterations >= m_ulMaxIterations)); //|| (m_fTemprature < 0.1) if temprature might be negative
}

bool CCOPCrossEntropy::RestartCritirionReached()
{
	//No restarts
	return false;
}

void CCOPCrossEntropy::Restart(COP::SolutionVector& NewInitial)
{
	return;
}
#pragma endregion
#pragma endregion

#pragma region float Local Search
#pragma region CFloatSimulatedAnnealingSearch
///
/// A CFloatSimulatedAnnealingSearch class - A simulated annealing search with floats , inherits from CSimulatedAnnealingSearch
/// This concrete class:
/// 1. Define the two template typenames : SolutionType as float and ValueType as float
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///
class CFloatSimulatedAnnealingSearch : public CSimulatedAnnealingSearch <float, float> 
{
public:
	
	virtual double Solve(float InitialSolution, 
					     double dTimeoutInSeconds, 
					     unsigned long ulMaxIterations,
					     float fInitialTemprature, 
					     float fTempratureStep,
					     float fNeighborhood,
					     unsigned long ulSeed);

	//virtual bool AcceptanceCritirionReached(ValueType);
	virtual void ChooseCandidate(float& Chosen);
	virtual void EvaluateSolution(const float& fSolution, float& fEvaluation);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual void Restart(float& fNewInitial); 

	float m_fNeighborhood;
};

double CFloatSimulatedAnnealingSearch::Solve(float InitialSolution, 
										     double dTimeoutInSeconds, 
										     unsigned long ulMaxIterations,
										     float fInitialTemprature, 
									         float fTempratureStep,
										     float fNeighborhood,
										     unsigned long ulSeed)
{
	m_fNeighborhood = fNeighborhood;

	return CSimulatedAnnealingSearch::Solve(InitialSolution, 
											dTimeoutInSeconds, 
											ulMaxIterations,
											fInitialTemprature, 
											fTempratureStep,
											ulSeed);
}

void CFloatSimulatedAnnealingSearch::ChooseCandidate(float& fChosen)
{
	//Choose StepSize uniformly in [-m_fNeighborhood,+m_fNeighborhood]
	std::uniform_real_distribution<float> fDistribution(-m_fNeighborhood,+m_fNeighborhood);
	float StepSize = fDistribution(AlgorithmRandomEngine);
	
	fChosen = m_CurrentSolution + StepSize;

	return;

}

void CFloatSimulatedAnnealingSearch::EvaluateSolution(const float& fSolution, float& fEvaluation)
{
	//minimize the function y=1-cos(2*pi*x)+0.1*x^2;
	fEvaluation = 1-5*cos(2*3.14159265358979323846264338f*fSolution)+0.1f*fSolution*fSolution;
	return;
	
}

bool CFloatSimulatedAnnealingSearch::StopCritirionReached()
{
	return (m_ulIterations >= m_ulMaxIterations) || (m_fTemprature < 0.1);
}

bool CFloatSimulatedAnnealingSearch::RestartCritirionReached()
{
	//No restarts
	return false;
}

void CFloatSimulatedAnnealingSearch::Restart(float& fNewInitial)
{
	fNewInitial = 0.0;
	return;
}
#pragma endregion

#pragma region  CFloatStochasticHillClimbingSearch
///
/// A CFloatHillClimbingSearch class - A simulated annealing search with integers , inherits from CSimulatedAnnealingSearch
/// This concrete class:
/// 1. Define the two template typenames : SolutionType as int* and ValueType as int
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///
class CFloatStochasticHillClimbingSearch : public CFixedTimeSearch <float, float> 
{
public:
	
	virtual double Solve(float InitialSolution, 
					     double dTimeoutInSeconds, 
					     unsigned long ulMaxIterations,
					     float fNeighborhood,
					     unsigned long ulSeed);
	virtual void ChooseCandidate(float& fChosen);
	virtual void EvaluateSolution(const float& fSolution, float& fEvaluation);
	virtual bool AcceptanceCritirionReached(const float& Solution);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual void Restart(float& fNewInitial); 

	float m_fNeighborhood;
	
};

double CFloatStochasticHillClimbingSearch::Solve(float InitialSolution, 
									             double dTimeoutInSeconds, 
												 unsigned long ulMaxIterations,
												 float fNeighborhood,
												 unsigned long ulSeed)
{
	m_fNeighborhood = fNeighborhood;

	return CFixedTimeSearch::Solve(InitialSolution, 
								dTimeoutInSeconds, 
								ulMaxIterations,
								ulSeed);		   
}

void CFloatStochasticHillClimbingSearch::ChooseCandidate(float& fChosen)
{
	//Choose StepSize uniformly in [-m_fNeighborhood,+m_fNeighborhood]
	std::uniform_real_distribution<float> fDistribution(-m_fNeighborhood,+m_fNeighborhood);
	float StepSize = fDistribution(AlgorithmRandomEngine);

	fChosen = m_CurrentSolution + StepSize;

	return;
}

void CFloatStochasticHillClimbingSearch::EvaluateSolution(const float& fSolution, float& fEvaluation)
{
	//minimize the function y=1-cos(2*pi*x)+0.1*x^2;
	 fEvaluation = 1-5*cos(2*3.14159265358979323846264338f*fSolution)+0.1f*fSolution*fSolution;
	 return;
}

bool CFloatStochasticHillClimbingSearch::AcceptanceCritirionReached(const float& Solution) 
{
	float CandidateValue; 
	CountedEvaluateSolution(Solution, CandidateValue);
	//Minimization Problem
	return (CandidateValue < m_ValueOfCurrentSolution);
}

bool CFloatStochasticHillClimbingSearch::StopCritirionReached()
{
	return (m_ulIterations >= m_ulMaxIterations);
}

bool CFloatStochasticHillClimbingSearch::RestartCritirionReached()
{
	//No restarts
	return false;
}

void CFloatStochasticHillClimbingSearch::Restart(float& fNewInitial)
{
	fNewInitial = 0.0;
	return;
}
#pragma endregion

#pragma region CFloatTabuSearch
///
/// A CFloatTabuSearch class - A tabu search with floats , inherits from CTabuSearch
/// This concrete class:
/// 1. Define the two template typenames : SolutionType as int* and ValueType as int
/// 2. Derive and implement the following functions:
///    a. ChooseCandidate()
///    b. EvaluateSolution()
///    d. StopCritirionReached()
///    e. RestartCritirionReached();
///    f. Restart()
///
class CFloatTabuSearch : public CTabuSearch <float, float> 
{
public:
	virtual void GenerateNeighbors();
	virtual void EvaluateSolution(const float& Solution, float& fEvaluation);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual void Restart(float& NewInitial); 
	virtual bool TabuForbidsSolution(float Solution); 
	virtual void TabuUpdate(float Solution); 

};

void CFloatTabuSearch::GenerateNeighbors() 
{
	float fMinStepSize = 0.5f;
	m_ulNeighborsCount = 0;
	unsigned long ulNeighbor = 0;

	while (ulNeighbor<MAX_NEIGHBORS_COUNT)
	{
		m_CurrentNeighbors[ulNeighbor++] = m_CurrentSolution + fMinStepSize*(ulNeighbor+1);

		if (ulNeighbor==MAX_NEIGHBORS_COUNT)
			break;

		m_CurrentNeighbors[ulNeighbor++] = m_CurrentSolution - fMinStepSize*(ulNeighbor);
	}
	m_ulNeighborsCount = ulNeighbor;
}

bool CFloatTabuSearch::TabuForbidsSolution(float Solution)
{
	//Solution is simple enough to be the attribute
	bool bForbiddenSolution = TabuContainsAttribute(Solution);
	return bForbiddenSolution;
}

void CFloatTabuSearch::TabuUpdate(float Solution)
{
	//Solution is simple enough to the attribute
	TabuPushAttribute(Solution);
}

void CFloatTabuSearch::EvaluateSolution(const float& fSolution, float& fEvaluation)
{
	//minimize the function y=1-cos(2*pi*x)+0.1*x^2;
	fEvaluation =  1-5*cos(2*3.14159265358979323846264338f*fSolution)+0.1f*fSolution*fSolution;
	return;
}

bool CFloatTabuSearch::StopCritirionReached()
{
	return (m_ulIterations >= m_ulMaxIterations);
}

bool CFloatTabuSearch::RestartCritirionReached()
{
	return false;
}

void CFloatTabuSearch::Restart(float& fNewInitial)
{
	fNewInitial = 0.0;
	return;
}

#pragma endregion
#pragma endregion

// great
//class CCOPGreatDelugeSearch : public CFixedTimeSearch <COP::SolutionVector, COP::GradesVector>
//{
//public:
//	CCOPGreatDelugeSearch(const string strPostFileName);
//
//	// to override
//	virtual double Solve(const COP::SolutionVector& InitialSolution, // everything from main
//		COP& Cop,
//		double dTimeoutInSeconds,
//		unsigned long ulMaxIterations,
//		unsigned int uiNumOfVarChangesInNeighborhood,
//		double m_Level,
//		unsigned long ulSeed);
//
//	// to override
//	virtual void ChooseCandidate(COP::SolutionVector& Chosen);
//	virtual void EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation);
//	virtual bool AcceptanceCritirionReached(const COP::SolutionVector& Solution);
//	virtual bool StopCritirionReached();
//	virtual bool RestartCritirionReached();
//	virtual void Restart(COP::SolutionVector& NewwInitial);
//
//	// helper funcs
//	double ScalarOfVec(const COP::SolutionVector& Solution) const;
//	// problem inputs
//	COP* m_Cop;
//	unsigned int m_uiNumOfVarChangesInNeighborhood;
//	double m_Level;
//	double m_beta;
//	double m_EstimatedQuality;
//};
//
//CCOPGreatDelugeSearch::CCOPGreatDelugeSearch(const string strPostFileName)
//	: CFixedTimeSearch(strPostFileName)
//{
//
//}
//
//double CCOPGreatDelugeSearch::ScalarOfVec(const COP::SolutionVector& Solution) const // sum of util's with respect to weights? 
//{
//	double dScalarValue = 0;
//
//	for (int iGradeIdx = 0; iGradeIdx < COP::MAX_LENGTH_OF_GRADES_VECTOR; iGradeIdx--)
//	{
//		if (Solution.aulSolutionVector[iGradeIdx] == 0) continue;
//
//		dScalarValue += Solution.aulSolutionVector[iGradeIdx];
//	}
//
//	return dScalarValue;
//
//}
//
//void CCOPGreatDelugeSearch::ChooseCandidate(COP::SolutionVector& Chosen) {
//	m_Cop->GenerateSingleNeighbor(Chosen, m_CurrentSolution, m_uiNumOfVarChangesInNeighborhood, AlgorithmRandomEngine);
//}
//
//void CCOPGreatDelugeSearch::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation) {
//	m_Cop->EvaluateSolution(Solution, Evaluation);
//}
//
//bool CCOPGreatDelugeSearch::AcceptanceCritirionReached(const COP::SolutionVector& Solution) {
//	COP::GradesVector ValueOfSolution;
//	CountedEvaluateSolution(Solution, ValueOfSolution);
//	if (ValueOfSolution < m_ValueOfCurrentSolution) {
//		return true;
//	}
//	else {
//		if (ScalarOfVec(Solution) <= m_Level) {
//			m_CurrentSolution = Solution;
//			m_ValueOfCurrentSolution = ValueOfSolution;
//			return false;
//		}
//	}
//	m_Level -= m_beta;
//	return false;
//}
//
//bool CCOPGreatDelugeSearch::StopCritirionReached() {
//	return (m_ulIterations >= m_ulMaxIterations);
//}
//
//bool CCOPGreatDelugeSearch::RestartCritirionReached() {
//	return false;
//}
//
//void CCOPGreatDelugeSearch::Restart(COP::SolutionVector& NewwInitial) {
//	for (unsigned int i = 0; i < COP::MAX_NUM_OF_VARS; i++) {
//		NewwInitial.aulSolutionVector[i] = 0;
//	}
//}
//
//double CCOPGreatDelugeSearch::Solve(const COP::SolutionVector& InitialSolution,
//	COP& Cop,
//	double dTimeoutInSeconds,
//	unsigned long ulMaxIterations,
//	unsigned int uiNumOfVarChangesInNeighborhood,
//	double Level,
//	unsigned long ulSeed)
//{
//	//*m_Cop = Cop;
//	m_Cop = &Cop;
//	m_uiNumOfVarChangesInNeighborhood = uiNumOfVarChangesInNeighborhood;
//	m_Level = Level;
//
//	//init beta
//	//COP::GradesVector ValueOfSolution;
//	//m_Cop->EvaluateSolution(InitialSolution, ValueOfSolution);
//	//m_EstimatedQuality = (float)ValueOfSolution.Scalarization();
//	m_EstimatedQuality = ScalarOfVec(InitialSolution);
//	m_beta = m_EstimatedQuality / ulMaxIterations; // calculate beta according to random initial solution
//
//
//	return CFixedTimeSearch::Solve(InitialSolution,
//		dTimeoutInSeconds,
//		ulMaxIterations,
//		ulSeed);
//}

class CCOPGreatDelugeSearch : public CFixedTimeSearch <COP::SolutionVector, COP::GradesVector>
{
public:
	CCOPGreatDelugeSearch(const string strPostFileName);

	// to override
	virtual double Solve(const COP::SolutionVector& InitialSolution, // everything from main
						COP& Cop,
						double dTimeoutInSeconds, 
						unsigned long ulMaxIterations,
						unsigned int uiNumOfVarChangesInNeighborhood,
						const COP::SolutionVector& svFinalScore, //  can be approximated with hill-climbing algo for example
						unsigned long ulSeed);

	// to override
	virtual void ChooseCandidate(COP::SolutionVector& Chosen);
	virtual void EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation);
	virtual bool AcceptanceCritirionReached(const COP::SolutionVector& Solution);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual void Restart(COP::SolutionVector& NewwInitial);
	
	// helper funcs
	double ScalarOfVec(const COP::SolutionVector& Solution) const; // just to test stuff (easier without infinite numbers)
	float CalcDeltaBeta(const COP::GradesVector& Sol1, const COP::GradesVector& Sol2);

	// problem inputs
	COP* m_Cop;
	unsigned int m_uiNumOfVarChangesInNeighborhood;
	float m_DeltaBeta;
	COP::GradesVector m_Beta;
	COP::GradesVector m_FinalScore; // just a variable for convinience holding the best estimated value of solution (i.e from hillClimb algo)
};

CCOPGreatDelugeSearch::CCOPGreatDelugeSearch(const string strPostFileName)
	: CFixedTimeSearch(strPostFileName)
{

}

double CCOPGreatDelugeSearch::ScalarOfVec(const COP::SolutionVector& Solution) const // sum of util's with respect to weights? 
{
	double dScalarValue = 0;

	for (int iGradeIdx = 0; iGradeIdx < COP::MAX_LENGTH_OF_GRADES_VECTOR; iGradeIdx--)
	{
		if (Solution.aulSolutionVector[iGradeIdx] == 0) continue;

		dScalarValue += Solution.aulSolutionVector[iGradeIdx];
	}

	return dScalarValue;

}

void CCOPGreatDelugeSearch::ChooseCandidate(COP::SolutionVector& Chosen) {
	m_Cop->GenerateSingleNeighbor(Chosen, m_CurrentSolution, m_uiNumOfVarChangesInNeighborhood, AlgorithmRandomEngine);
}

void CCOPGreatDelugeSearch::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation) {
	m_Cop->EvaluateSolution(Solution, Evaluation);
}

bool CCOPGreatDelugeSearch::AcceptanceCritirionReached(const COP::SolutionVector& Solution) {
	COP::GradesVector ValueOfSolution;
	CountedEvaluateSolution(Solution, ValueOfSolution);
	if (ValueOfSolution <= m_ValueOfCurrentSolution) { // i think it should be ->m_ValueOfBestSolution rather then m_ValueOfCurrentSolution according to the pseudocode
		return true;
	}
	else if (ValueOfSolution <= m_Beta) {
			return true;
		}
	m_Beta = m_Beta - m_DeltaBeta;
	return false;
}

float CCOPGreatDelugeSearch::CalcDeltaBeta(const COP::GradesVector& Sol1, const COP::GradesVector& Sol2) // difference between two solutions for each entry
{
	float temp = Sol1 - Sol2;
	return temp / m_ulMaxIterations;
}

bool CCOPGreatDelugeSearch::StopCritirionReached() {
	return (m_ulIterations >= m_ulMaxIterations);
}

bool CCOPGreatDelugeSearch::RestartCritirionReached() {
	return false;
}

void CCOPGreatDelugeSearch::Restart(COP::SolutionVector& NewwInitial) {
	for (unsigned int i = 0; i < COP::MAX_NUM_OF_VARS; i++) {
		NewwInitial.aulSolutionVector[i] = 0;
	}
}

double CCOPGreatDelugeSearch::Solve(const COP::SolutionVector& InitialSolution,
								 COP& Cop,
								 double dTimeoutInSeconds,
								 unsigned long ulMaxIterations,
								 unsigned int uiNumOfVarChangesInNeighborhood,
								 const COP::SolutionVector& svFinalScore,
								 unsigned long ulSeed)
{
	//*m_Cop = Cop;
	m_Cop = &Cop;
	m_uiNumOfVarChangesInNeighborhood = uiNumOfVarChangesInNeighborhood;

	//init beta
	m_Cop->EvaluateSolution(InitialSolution, m_Beta); // contains B_0 in m_Beta
	m_Cop->EvaluateSolution(svFinalScore, m_FinalScore);
	m_DeltaBeta = CalcDeltaBeta(m_Beta, m_FinalScore); // calculate beta according to random svFinalScore
													   // svFinalScore can be estimated with hillClimb algorithm before hand


	return CFixedTimeSearch::Solve(InitialSolution,
								   dTimeoutInSeconds,
								   ulMaxIterations,
								   ulSeed);
}

//GREAT_DELUGE 124 asda 1.0 10 331991908 -neighborhood 10 -initlvl 0.3
//LJ 124 asdas 1.0 10 331991908 -neighboargood 5.0
class Float_Luus_Jaakola : public CFixedTimeSearch <float, float>
{
public:

	Float_Luus_Jaakola(const string strPostFileName);

	virtual double Solve(const float& InitialSolution,
						double dTimeoutInSeconds,
						unsigned long ulMaxIterations,
						float fNeighborhood,
						unsigned long ulSeed);

	virtual void ChooseCandidate(float& Chosen);
	virtual void EvaluateSolution(const float& Solution, float& Evaluation);
	virtual bool AcceptanceCritirionReached(const float& Solution);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual void Restart(float& NewInitial);

	float m_fNeighborhood;
};

Float_Luus_Jaakola::Float_Luus_Jaakola(const string strPostFileName)
	: CFixedTimeSearch(strPostFileName)
{

}

bool Float_Luus_Jaakola::StopCritirionReached() {
	return (m_ulIterations >= m_ulMaxIterations) || (m_fNeighborhood < 0.01);
}

bool Float_Luus_Jaakola::RestartCritirionReached()
{
	//No restarts
	return false;
}

void Float_Luus_Jaakola::Restart(float& NewInitial)
{
	NewInitial = 0.0;
	return;
}

void Float_Luus_Jaakola::EvaluateSolution(const float& Solution, float& Evaluation) {
	// took this function from the other float classes
	Evaluation = 1 - 5 * cos(2 * 3.14159265358979323846264338f * Solution) + 0.1f * Solution * Solution;
	return;
}

bool Float_Luus_Jaakola::AcceptanceCritirionReached(const float& Solution) {
	float CandidateValue;
	CountedEvaluateSolution(Solution, CandidateValue);

	if (CandidateValue < m_ValueOfCurrentSolution) {
		return true;
	}

	m_fNeighborhood *= (float)0.95;
	return false;

}

void Float_Luus_Jaakola::ChooseCandidate(float& chosen) {
	std::uniform_real_distribution<float> fDistribution(-m_fNeighborhood * 2, m_fNeighborhood * 2);
	float StepSize = fDistribution(AlgorithmRandomEngine);

	chosen = m_CurrentSolution + StepSize;

	return;
}

double Float_Luus_Jaakola::Solve(const float& InitialSolution,
						 double dTimeoutInSeconds,
						 unsigned long ulMaxIterations,
						 float fNeighborhood,
						 unsigned long ulSeed) 
{

	m_fNeighborhood = fNeighborhood;

	return CFixedTimeSearch::Solve(InitialSolution,
								  dTimeoutInSeconds,
								  ulMaxIterations,
								  ulSeed);
}


typedef struct EliteSolutionVector {
	COP::SolutionVector Solution;
	COP::GradesVector Grade;
}EliteSolutionVector;

class CCOPStochasticLocalBeamSearch : public CFixedTimeSearch <COP::SolutionVector, COP::GradesVector>
{
public:
	static const unsigned int MAX_ELITE_SIZE = 30;

	CCOPStochasticLocalBeamSearch(const string strPostFileName, unsigned int uiElites);

	~CCOPStochasticLocalBeamSearch();

	virtual double Solve(const COP::SolutionVector& InitialSolution,
						 COP& Cop,
						 double dTimeoutInSeconds,
						 unsigned long ulMaxIterations,
						 unsigned int uiNumOfVarChangesInNeighborhood,
						 unsigned long ulSeed);
						 
	virtual void ChooseCandidate(COP::SolutionVector& Chosen);
	virtual void EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation);
	virtual bool AcceptanceCritirionReached(const COP::SolutionVector& Solution);
	virtual bool StopCritirionReached();
	virtual bool RestartCritirionReached();
	virtual void Restart(COP::SolutionVector& NewInitial);

	//Helper functions
	void GenerateSolutionsFromElites();
	void GenerateSolutionsFromInitial(const COP::SolutionVector& Initial);
	//void SortElites(); // in case we want to remove the Stochastic element

	//Problem input
	COP* m_Cop;
	unsigned int m_uiElites;
	unsigned int m_uiNumOfVarChangesInNeighborhood;
	EliteSolutionVector* m_CurrentElites;
	EliteSolutionVector* m_ElitesNeighbors;


};

CCOPStochasticLocalBeamSearch::CCOPStochasticLocalBeamSearch(const string strPostFileName, unsigned int uiElites)
	:CFixedTimeSearch(strPostFileName),
	 m_uiElites(uiElites)
{
	assert(m_uiElites <= MAX_ELITE_SIZE);
	m_CurrentElites = new EliteSolutionVector[m_uiElites];
	m_ElitesNeighbors = new EliteSolutionVector[MAX_ELITE_SIZE * m_uiElites];
}

CCOPStochasticLocalBeamSearch::~CCOPStochasticLocalBeamSearch()
{
	delete[] m_CurrentElites;
	delete[] m_ElitesNeighbors;
}

//inline int CompareElites(const void* SampleA, const void* SampleB) // in case we want to remove the Stochastic element
//{
//	EliteSolutionVector* eSampleA = (EliteSolutionVector*)SampleA;
//	EliteSolutionVector* eSampleB = (EliteSolutionVector*)SampleB;
//
//	if (eSampleA->Grade > eSampleB->Grade) return -1;
//	if (eSampleA->Grade < eSampleB->Grade) return 1;
//	return 0;
//}

//void CCOPStochasticLocalBeamSearch::SortElites() // in case we want to remove the Stochastic element
//{
//	qsort(m_ElitesNeighbors, m_uiElites * MAX_ELITE_SIZE, sizeof(EliteSolutionVector), CompareElites);
//}

void CCOPStochasticLocalBeamSearch::GenerateSolutionsFromElites() {
	for (unsigned int elite = 0; elite < m_uiElites; elite++) {
		for (unsigned int eliteNeighbor = 0; eliteNeighbor < MAX_ELITE_SIZE; eliteNeighbor++) {
			m_Cop->GenerateSingleNeighbor(m_ElitesNeighbors[elite * MAX_ELITE_SIZE + eliteNeighbor].Solution, m_CurrentElites[elite].Solution, m_uiNumOfVarChangesInNeighborhood, AlgorithmRandomEngine);
		}
	}

	for (unsigned int elite = 0; elite < m_uiElites; elite++) {
		for (unsigned int eliteNeighbor = 0; eliteNeighbor < MAX_ELITE_SIZE; eliteNeighbor++) {
			m_Cop->EvaluateSolution(m_ElitesNeighbors[elite * MAX_ELITE_SIZE + eliteNeighbor].Solution, m_ElitesNeighbors[elite * MAX_ELITE_SIZE + eliteNeighbor].Grade);
		}
	}

}
void CCOPStochasticLocalBeamSearch::GenerateSolutionsFromInitial(const COP::SolutionVector& Initial) {

	for (unsigned int i = 0; i < m_uiElites; i++) {
		m_Cop->GenerateSingleNeighbor(m_CurrentElites[i].Solution, Initial, m_uiNumOfVarChangesInNeighborhood, AlgorithmRandomEngine);
	}
	for (unsigned int i = 0; i < m_uiElites; i++) {
		m_Cop->EvaluateSolution(m_CurrentElites[i].Solution, m_CurrentElites[i].Grade);
	}

}
double CCOPStochasticLocalBeamSearch::Solve(const COP::SolutionVector& InitialSolution,
											COP& Cop,
											double dTimeoutInSeconds,
											unsigned long ulMaxIterations,
											unsigned int uiNumOfVarChangesInNeighborhood,
											unsigned long ulSeed) {

	m_Cop = &Cop;
	m_uiNumOfVarChangesInNeighborhood = uiNumOfVarChangesInNeighborhood;
	GenerateSolutionsFromInitial(InitialSolution);

	return CFixedTimeSearch::Solve(InitialSolution,
								   dTimeoutInSeconds,
								   ulMaxIterations,
								   ulSeed);

}
void CCOPStochasticLocalBeamSearch::ChooseCandidate(COP::SolutionVector& Chosen) {
	GenerateSolutionsFromElites();
	unsigned int elites_num = 0;
	//SortElites(); // in case we want to remove the Stochastic element
	for (unsigned int candidate = 0; candidate < MAX_ELITE_SIZE * m_uiElites; candidate++) {
		if (m_ElitesNeighbors[candidate].Grade <= m_ValueOfCurrentSolution && elites_num < m_uiElites) {
			m_CurrentElites[elites_num].Solution = m_ElitesNeighbors[candidate].Solution;
			m_CurrentElites[elites_num].Grade = m_ElitesNeighbors[candidate].Grade;
			elites_num++;
		}
	}

	std::uniform_int_distribution<uint32_t> UIntDistForChosen(0, m_uiElites - 1);

	// maybe if elite size is 0 then fill wil randoms?
	while (elites_num < m_uiElites) {
		unsigned int idx = UIntDistForChosen(AlgorithmRandomEngine);
		m_CurrentElites[elites_num].Solution = m_ElitesNeighbors[idx].Solution;
		m_CurrentElites[elites_num].Grade = m_ElitesNeighbors[idx].Grade;
		elites_num++;
	}


	//unsigned int elites_num = 0; // in case we want to remove the Stochastic element
	//while(elites_num != m_uiElites){
	//	m_CurrentElites[elites_num] = m_ElitesNeighbors[MAX_ELITE_SIZE * m_uiElites - 1 - elites_num];
	//	elites_num++;
	//}
	//for (auto i = 0; i < elites_num;i++) {
	//	cout << m_CurrentElites[i].Solution << endl;
	//	cout << endl;
	//}
	//Chosen = m_CurrentElites[0].Solution;
	Chosen = m_CurrentElites[UIntDistForChosen(AlgorithmRandomEngine)].Solution;
	
}

void CCOPStochasticLocalBeamSearch::EvaluateSolution(const COP::SolutionVector& Solution, COP::GradesVector& Evaluation) {
	m_Cop->EvaluateSolution(Solution, Evaluation);
}

bool CCOPStochasticLocalBeamSearch::AcceptanceCritirionReached(const COP::SolutionVector& Solution) {
	COP::GradesVector SolutionValue;
	CountedEvaluateSolution(Solution, SolutionValue);
	return (SolutionValue <= m_ValueOfCurrentSolution);
}
//
bool CCOPStochasticLocalBeamSearch::StopCritirionReached() {
	return (m_ulIterations >= m_ulMaxIterations);
}

bool CCOPStochasticLocalBeamSearch::RestartCritirionReached() {
	return false; // no restarts
}

void CCOPStochasticLocalBeamSearch::Restart(COP::SolutionVector& NewInitial) {
	return;
}