import numpy as np
import pandas as pd
import os

from structClasses import ValuesPerVars, M, MAX_NUM_OF_MS


class Reader:
    def __init__(self, path, problem_seed):
        self.path = path
        self.problem_seed = problem_seed
        self.valuesPerVariable = ValuesPerVars()
        self.binaryConstraintsMatrix = self.matrix_reader()
        self.maxValuesNum, self.MS = self.MS_maxValuesNum_reader()
        self.ValuesPerVariable_reader()

    def matrix_reader(self):
        files = os.listdir(self.path)
        for file in files:
            if file == "BinaryConstraintsMatrix_" + str(self.problem_seed) + ".txt":
                with open(self.path + file) as f:
                    bin_mat = np.array(f.read().split(), dtype=np.int16)
        return bin_mat

    def MS_maxValuesNum_reader(self):
        files = os.listdir(self.path)
        for file in files:
            if file == "Ms_" + str(self.problem_seed) + ".txt":
                with open(self.path + file) as f:
                    MS_file = np.array(f.read().split(), dtype=np.int16)

        maxValuesNum = MS_file[-2]

        MS_array_amounts = MS_file[:-2]
        MS_class_array = [M()] * MAX_NUM_OF_MS
        for i in range(MAX_NUM_OF_MS):
            MS_class_array[i].amount = MS_array_amounts[i]

        self.valuesPerVariable.validVarAmount = MS_file[-1]
        return maxValuesNum, MS_class_array

    def ValuesPerVariable_reader(self):
        files = os.listdir(self.path)
        vpvFile = None
        for file in files:
            if file == "ValuesPerVariable_" + str(self.problem_seed) + ".csv":
                vpvFile = pd.read_csv(self.path + file, dtype=np.int16)

        list_of_valid_indexes = list(set(vpvFile['index']))

        for idx in list_of_valid_indexes:
            valid_info_for_idx = vpvFile[vpvFile['index'] == idx]
            self.valuesPerVariable.varsData[idx].valuesB = np.array(valid_info_for_idx['B'], dtype=np.int16)
            self.valuesPerVariable.varsData[idx].valuesM = np.array(valid_info_for_idx['M'], dtype=np.int16)
            self.valuesPerVariable.varsData[idx].valuesP = np.array(valid_info_for_idx['P'], dtype=np.int16)
            self.valuesPerVariable.varsData[idx].valuesQ = np.array(valid_info_for_idx['Q'], dtype=np.int16)
            self.valuesPerVariable.varsData[idx].ucPrio = np.int16(valid_info_for_idx['ucPrio'].iloc[0])
            self.valuesPerVariable.varsData[idx].valuesAmount = np.int16(valid_info_for_idx['ulValuesAmount'].iloc[0])


if __name__ == '__main__':
    path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\Problems\\'
    r = Reader(path=path, problem_seed=500)
    r.ValuesPerVariable_reader()
    # print(type(r.binaryConstraintsMatrix), r.binaryConstraintsMatrix.shape)
    # print(r.binaryConstraintsMatrix[:20], type(r.binaryConstraintsMatrix[2]), r.binaryConstraintsMatrix[1], r.binaryConstraintsMatrix[12312])
    # print(r.maxValuesNum, r.MS)
    # print("valuespervariable info")
    # print(r.valuesPerVariable.varsData[0].valuesB)
    # print(r.valuesPerVariable.varsData[0].valuesP)
    # print(r.valuesPerVariable.varsData[0].valuesM)
    # print(r.valuesPerVariable.varsData[0].valuesQ)
    # print(r.valuesPerVariable.varsData[0].valuesAmount)
    # print(r.valuesPerVariable.varsData[0].ucPrio)
    # print(r.valuesPerVariable.varsData[17].ucPrio)








