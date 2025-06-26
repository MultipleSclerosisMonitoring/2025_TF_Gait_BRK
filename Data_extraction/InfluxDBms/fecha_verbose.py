import argparse
from datetime import datetime, timedelta

class VAction(argparse.Action):
    """
    Custom class to handle the verbosity argument.

    This class allows counting the number of times the `-v` or `--verbose` 
    argument is used, as well as receiving an optional numeric value. 
    

    Args:
        option_strings (list): list of option strings that correspond # to this action. 
        to this action.
        dest (str): Name of the attribute to which the value will be # assigned.
        nargs (int, optional): Number of arguments to consume. Default is None.
        const (any, optional): Constant value to use if nargs is `?`.
        default (any, optional): Default value to use if not given.
        type (type, optional): Type of argument expected.
        choices (list, optional): Valid options for the argument.
        required (bool, optional): If True, the argument is required.
        help (str, optional): Help description for the argument.
        metavar (str, optional): Name to use in the help message.

    Attributes:
        values (int): Verbosity level counter.
    
    
    
    """

    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        super(VAction, self).__init__(option_strings, dest, nargs, const,
                                      default, type, choices, required,
                                      help, metavar)
        self.values = 0

    def __call__(self, parser, args, values, option_string=None):

        """
        Processes the argument value and updates the verbosity counter.

        Args:
            parser (argparse.ArgumentParser): The parser object of argparse.
            args (Namespace): Namespace containing the arguments.
            values (any): Value of the argument.    
            option_string (str, optional): Option string that was used.
        
        """
        if values is None:
            self.values += 1
        else:
            try:
                self.values = int(values)
            except ValueError:
                self.values = values.count('v') + 1
        setattr(args, self.dest, self.values)

class BatchProcess:

    """
    Main class for handling arguments and running batch processes.

    This class handles the configuration of the input arguments and the 
    execution of the main logic of the batch process.
    

    Attributes:
        config (argparse.Namespace): Namespace containing the parsed arguments. 
    
    """

    def __init__(self, args):

        """
        Initialises the class and parses the input arguments.
        """
        self.args = args

    def get_cnf(self):

        """
        Function returning the fiel where the relevant configuration is placed.

        :return: Path to the config file
        :rtype: str
        """
        return self.args.path

    def run(self):

        """
        Runs the process in batch.

        Retrieves the input parameters and executes the main logic  of the process.
        """
        verbose = self.args.verbose
        if verbose >= 1:
            print(f"Running BatchProcess with verbosity: {verbose}")
        self.main_process()

    def main_process(self):

        """
        Simulates the main execution process.

        This method contains the main logic of the process and may include 
        additional operations as required.
        """
        verbose = self.args.verbose
        if verbose >= 1:
            print("Starting main process...")
        for i in range(3):
            if verbose >= 2:
                print(f"Iteration {i} in the main process.")
            if verbose >= 3:
                print(f"Internal details in iteration {i}: {{'key': 'value'}}")
        if verbose >= 1:
            print("Main process completed.")
