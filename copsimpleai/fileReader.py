import numpy as np
import pandas as pd
import os

from structClasses import ValuesPerVars, M, MAX_NUM_OF_MS

"""
class responsible for reading a COP problem Created into appropriate fields.
"""


class Reader:
    def __init__(self, path, problem_seed):
        """Creates Reader class for reading COP problem of for a given problem seed.

        Args:
            path (str): path to the folder which contains all the files needed to initialize a COP problem created.
            problem_seed (int): COP problem seed number.

        Returns:
             None.
        """
        self.path = path
        self.problem_seed = problem_seed
        self.valuesPerVariable = ValuesPerVars()
        self.binaryConstraintsMatrix = self.matrix_reader()
        self.maxValuesNum, self.MS = self.MS_maxValuesNum_reader()
        self.ValuesPerVariable_reader()

    def matrix_reader(self):
        """Reading the BinaryConstraintsMatrix_ text file for the problem seed
         and returning it as numpy 1d array of type int.

        Args:
            None.

        Returns:
             list[int]: constrains matrix represented as 1d array. same as in localsearch.h file.
        """
        files = os.listdir(self.path)
        for file in files:
            if file == "BinaryConstraintsMatrix_" + str(self.problem_seed) + ".txt":
                with open(self.path + file) as f:
                    bin_mat = np.array(f.read().split(), dtype=np.int16)
        return bin_mat

    def MS_maxValuesNum_reader(self):
        """Reading Ms_ text file for the problem seed and initializing req fields.
        all according to the initialization in localsearch.h file from CPP.

        Args:
            None.

        Returns:
            (int, list[M]): maximum value number, array of M classes of length MAX_NUM_OF_MS already initialized.
        """
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
        """Reading ValuesPerVariable_ cvs file as DataFrame.
        Initializing valuesPerVariable Data with numpy 1d arrays of type int where needed and ints otherwise.

        Args:
            None.

        Returns:
            None.
        """
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









