import sys, os
import importlib


class InternalCoordinateManager:
    def __init__(self,
                 int_function,
                 int_directory,
                 python_file,
                 int_names):
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
