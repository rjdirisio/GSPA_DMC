import sys, os
import importlib


class InternalCoordinateManager:
    def __init__(self,
                 int_function,
                 int_directory,
                 python_file,
                 int_names):
        """

        :param int_function: The function in the example.py file you created that returns the 3n-6 coordinates of all walkers
        :param int_directory: The directory in which example.py lives in
        :param python_file: The python file 'example.py'
        :param int_names: A list of strings describing the internal coordinates. Will be used for assignments.txt
        """
        self.int_func = int_function
        self.pyFile = python_file
        self.int_dir = int_directory
        self.int_names = int_names
        self._curdir = os.getcwd()
        self._initialize()

    def get_int_names(self):
        return self.int_names

    def _initialize(self):
        if self.int_dir == '':
            pass
        else:
            os.chdir(self.int_dir)
        sys.path.insert(0, os.getcwd())
        module = self.pyFile.split(".")[0]
        x = importlib.import_module(module)
        self._int = getattr(x, self.int_func)
        os.chdir(self._curdir)

    def get_ints(self, cds):
        return self._int(cds)
