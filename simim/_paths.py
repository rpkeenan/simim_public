import os
import numpy as np
from importlib_resources import files
from .siminterface._sims import _checksim

class _paths():
    """Class to handle the paths to simulations"""

    def __init__(self):
        """Load the paths files"""

        # Absolute path to simim package
        simim_path = files('simim')
        if not os.path.exists(simim_path.joinpath('resources')):
            os.mkdir(simim_path.joinpath('resources'))
        
        # Path to the list of paths
        self.root_file = simim_path.joinpath('resources','rootpath.txt')
        self.lc_file = simim_path.joinpath('resources','lcpaths.txt')
        self.path_file = simim_path.joinpath('resources','filepahts.txt')
        self.sfr_file = simim_path.joinpath('resources','sfrpaths.txt')

        # Load values, if they exist
        self.root = None
        if os.path.exists(self.root_file):
            with open(self.root_file) as file:
                self.root = file.readlines()[0].replace('\n','')

        self.paths = {}
        if os.path.exists(self.path_file):
            with open(self.path_file) as file:
                for line in file.readlines():
                    line_split = line.replace('\n','').split(' ')
                    self.paths[line_split[0]] = line_split[1]

        self.lightcones = {}
        if os.path.exists(self.lc_file):
            with open(self.lc_file) as file:
                for line in file.readlines():
                    line_split = line.replace('\n','').split(' ')
                    self.lightcones[line_split[0]] = line_split[1]

        self.sfrs = {}
        if os.path.exists(self.sfr_file):
            with open(self.sfr_file) as file:
                for line in file.readlines():
                    line_split = line.replace('\n','').split(' ')
                    self.sfrs[line_split[0]] = line_split[1]

    def _setuppath(self,root='~'):
        """Create a directory root/simim_resources/simulations"""

        # Get the right thing for the home directory
        if root == '~':
            root = os.path.expanduser("~")

        # Check if a root already exists and whehter it should be replaced
        if self.root:
            if root != self.root:
                print("A root has already been specified:")
                print("   {}".format(self.root))
                answer = input("Do you want to set a new root? y/n: ")

                while answer != 'y':
                    if answer == 'n':
                        print("Keeping existing root")
                        root = self.root
                        break
                    print("Response not recognized.")
                    answer = input("Do you want to set a new root? y/n: ")

        # Confirm location of root
        if not os.path.exists(root):
            raise NameError("Specified path not found")

        root = os.path.abspath(root)
        print("Files will be saved in {}".format(os.path.join(root,'simim_resources')))
        answer = input("Is this okay? y/n: ")
        while answer != 'y':
            if answer == 'n':
                print("Aborting root setup")
                return
            print("Response not recognized.\n")
            print("Files will be saved in {}".format(os.path.join(root,'simim_resources')))
            answer = input("Is this okay? y/n: ")

        # Save the root location
        with open(self.root_file,'w') as file:
            file.write(root)
        self.root = root

        if not os.path.exists(os.path.join(root,'simim_resources')):
            os.mkdir(os.path.join(root,'simim_resources'))
        if not os.path.exists(os.path.join(root,'simim_resources','simulations')):
            os.mkdir(os.path.join(root,'simim_resources','simulations'))
        if not os.path.exists(os.path.join(root,'simim_resources','lightcones')):
            os.mkdir(os.path.join(root,'simim_resources','lightcones'))
        if not os.path.exists(os.path.join(root,'simim_resources','sfrs')):
            os.mkdir(os.path.join(root,'simim_resources','sfrs'))

    def _newsimpath(self,sim,new_path='auto',checkoverwrite=True):
        """Add a new path to the paths list"""

        # Check that simulation is supported
        _checksim(sim)

        # Check if a path is already specified
        if checkoverwrite:
            if sim in self.paths.keys():
                print("A path for this simulation has already been specified:")
                print("   {}".format(self.paths[sim]))
                answer = input("Do you want to replace this file? y/n: ")

                while answer != 'y':
                    if answer == 'n':
                        print("Keeping existing path")
                        return
                    print("Response not recognized.")
                    answer = input("Do you want to replace this file? y/n: ")

        # Get the new path and add to the list
        if not new_path:
            new_path = input("Please specify a path for {} data: ".format(sim))
            if not os.path.exists(new_path):
                raise NameError("Specified path does not exist. Please create path and try again.")
        elif new_path == 'auto':
            if not self.root:
                raise NameError("Root not specified, auto-path will fail")
            new_path = os.path.join(self.root,'simim_resources','simulations',sim)
            if not os.path.exists(new_path):
                os.mkdir(new_path)

        new_path = os.path.abspath(new_path)

        # Write the new path name
        with open(self.path_file,'a') as file:
            file.write('{} {}\n'.format(sim,new_path))

        self.paths[sim] = new_path

    def _newlcpath(self,sim,new_path='auto',checkoverwrite=True):
        """Add a new path to the lightcones list"""

        # Check that simulation is supported
        _checksim(sim)

        # Check if a path is already specified
        if checkoverwrite:
            if sim in self.lightcones.keys():
                print("A location for light cones from this simulation has already been specified:")
                print("   {}".format(self.lightcones[sim]))
                answer = input("Do you want to replace this file? y/n: ")

                while answer != 'y':
                    if answer == 'n':
                        print("Keeping existing path")
                        return
                    print("Response not recognized.")
                    answer = input("Do you want to replace this file? y/n: ")

        # Get the new path and add to the list
        if not new_path:
            new_path = input("Please specify a path for {} data: ".format(sim))
            if not os.path.exists(new_path):
                raise NameError("Specified path does not exist. Please create path and try again.")
        elif new_path == 'auto':
            if not self.root:
                raise NameError("Root not specified, auto-path will fail")
            new_path = os.path.join(self.root,'simim_resources','lightcones',sim)
            if not os.path.exists(new_path):
                os.mkdir(new_path)

        new_path = os.path.abspath(new_path)

        # Write the new path name
        with open(self.lc_file,'a') as file:
            file.write('{} {}\n'.format(sim,new_path))

        self.lightcones[sim] = new_path


    def _newsfrpath(self,item,new_path='auto',checkoverwrite=True):
        """Add a new path to the sfr list"""

        # Check if a path is already specified
        if checkoverwrite:
            if item in self.sfrs.keys():
                print("A location for this SFR data has already been specified:")
                print("   {}".format(self.sfrs[item]))
                answer = input("Do you want to replace this file? y/n: ")

                while answer != 'y':
                    if answer == 'n':
                        print("Keeping existing path")
                        return
                    print("Response not recognized.")
                    answer = input("Do you want to replace this file? y/n: ")

        # Get the new path and add to the list
        if not new_path:
            new_path = input("Please specify a path for {} data: ".format(item))
            if not os.path.exists(new_path):
                raise NameError("Specified path does not exist. Please create path and try again.")
        elif new_path == 'auto':
            if not self.root:
                raise NameError("Root not specified, auto-path will fail")
            new_path = os.path.join(self.root,'simim_resources','sfrs',item)
            if not os.path.exists(new_path):
                os.mkdir(new_path)

        new_path = os.path.abspath(new_path)

        # Write the new path name
        with open(self.sfr_file,'a') as file:
            file.write('{} {}\n'.format(item,new_path))

        self.sfrs[item] = new_path

class setupsimim():
    """Does all of the initialization"""

    def __init__(self):

        path = _paths()

        print("Please specify a path to save data directories.")
        print("Specifying no path will set the path to your home direcotry.")
        root = input("Path: ")
        if root == '':
            root = '~'
        elif not os.path.exists(root):
            raise NameError("Specified path does not exist. Please create path and try again.")
        path._setuppath(root=root)