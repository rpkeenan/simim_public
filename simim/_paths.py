import os
from importlib_resources import files

def _checkload_root_file(path):
    """Check if root path file exists and read file contents"""
    endpoint = None
    if os.path.exists(path):
        with open(path) as file:
            endpoint = file.readlines()[0].replace('\n','')
    return endpoint

def _checkload_path_file(path):
    """Check if a path listing file exists and read file contents"""
    endpoint = {}
    if os.path.exists(path):
        with open(path) as file:
            for line in file.readlines():
                line_split = line.replace('\n','').split(' ')
                endpoint[line_split[0]] = line_split[1]
    return endpoint

class _SimIMPaths():
    """Class to handle the paths to simulations"""

    def __init__(self):
        """Load the paths files"""

        # Absolute path to simim package
        simim_path = files('simim')
        if not os.path.exists(simim_path.joinpath('resources')):
            os.mkdir(simim_path.joinpath('resources'))
        
        # Path to the file where the data directory is listed
        self.root_file = simim_path.joinpath('resources','rootpath.txt')
        self.root = _checkload_root_file(self.root_file)

        # Places where files are saved...
        self.lcpath_ext = os.path.join('simim_resources','.paths','lcpaths.txt')
        self.simpath_ext = os.path.join('simim_resources','.paths','simpaths.txt')
        self.proppath_ext = os.path.join('simim_resources','.paths','proppaths.txt')

        self.lc_ext = os.path.join('simim_resources','lightcones')
        self.sim_ext = os.path.join('simim_resources','simulations')
        self.prop_ext = os.path.join('simim_resources','galprops')

        self._load_paths()

    def _load_paths(self):
        if self.root is None:
            self.lcs = {}
            self.sims = {}
            self.props = {}

        else:
            self.lcs = _checkload_path_file(os.path.join(self.root,self.lcpath_ext))
            self.sims = _checkload_path_file(os.path.join(self.root,self.simpath_ext))
            self.props = _checkload_path_file(os.path.join(self.root,self.proppath_ext))
        
        # # Path to the list of paths
        # self.lc_file = simim_path.joinpath('resources','lcpaths.txt')
        # self.path_file = simim_path.joinpath('resources','filepaths.txt')
        # self.sfr_file = simim_path.joinpath('resources','sfrpaths.txt')

        # # Load values, if they exist
        # self.root = None
        # if os.path.exists(self.root_file):
        #     with open(self.root_file) as file:
        #         self.root = file.readlines()[0].replace('\n','')

        # self.paths = {}
        # if os.path.exists(self.path_file):
        #     with open(self.path_file) as file:
        #         for line in file.readlines():
        #             line_split = line.replace('\n','').split(' ')
        #             self.paths[line_split[0]] = line_split[1]

        # self.lightcones = {}
        # if os.path.exists(self.lc_file):
        #     with open(self.lc_file) as file:
        #         for line in file.readlines():
        #             line_split = line.replace('\n','').split(' ')
        #             self.lightcones[line_split[0]] = line_split[1]

        # self.sfrs = {}
        # if os.path.exists(self.sfr_file):
        #     with open(self.sfr_file) as file:
        #         for line in file.readlines():
        #             line_split = line.replace('\n','').split(' ')
        #             self.sfrs[line_split[0]] = line_split[1]

    def _setuppath(self,root='~',confirm_with_user=False):
        """Create a directory root/simim_resources/simulations"""

        # Get the right thing for the home directory
        if root == '~':
            root = os.path.expanduser("~")

        # Check if a root already exists and whehter it should be replaced
        if self.root is not None:
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

        if confirm_with_user:
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

        for path in [os.path.join(root,'simim_resources'),
                      os.path.join(root,'simim_resources','.paths'),
                      os.path.join(root,self.sim_ext),
                      os.path.join(root,self.lc_ext),
                      os.path.join(root,self.prop_ext)]:
            if not os.path.exists(path):
                os.mkdir(path)

        self._load_paths()

    def _newsimpath(self,sim,new_path='auto',checkoverwrite=True):
        """Add a new path to the paths list"""

        # Check that a root location is known
        if self.root is None:
            raise ValueError("A data directory for SimIM has not been specified - please import and run simim.setupsimim")

        # Check if a path is already specified
        if checkoverwrite:
            if sim in self.sims:
                print("A path for this simulation has already been specified:")
                print("   {}".format(self.sims[sim]))
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
            new_path = os.path.join(self.root,self.sim_ext,sim)
            if not os.path.exists(new_path):
                os.mkdir(new_path)

        new_path = os.path.abspath(new_path)

        # Write the new path name
        with open(os.path.join(self.root,self.simpath_ext),'a') as file:
            file.write('{} {}\n'.format(sim,new_path))

        self.sims[sim] = new_path

    def _newlcpath(self,sim,new_path='auto',checkoverwrite=True):
        """Add a new path to the lcs list"""

        # Check that a root location is known
        if self.root is None:
            raise ValueError("A data directory for SimIM has not been specified - please import and run simim.setupsimim")

        # Check if a path is already specified
        if checkoverwrite:
            if sim in self.lcs:
                print("A location for light cones from this simulation has already been specified:")
                print("   {}".format(self.lcs[sim]))
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
            new_path = os.path.join(self.root,self.lc_ext,sim)
            if not os.path.exists(new_path):
                os.mkdir(new_path)

        new_path = os.path.abspath(new_path)

        # Write the new path name
        with open(os.path.join(self.root,self.lcpath_ext),'a') as file:
            file.write('{} {}\n'.format(sim,new_path))

        self.lcs[sim] = new_path

    def _newproppath(self,item,new_path='auto',checkoverwrite=True):
        """Add a new path to the props list"""

        # Check that a root location is known
        if self.root is None:
            raise ValueError("A data directory for SimIM has not been specified - please import and run simim.setupsimim")

        # Check if a path is already specified
        if checkoverwrite:
            if item in self.props:
                print("A location for this SFR data has already been specified:")
                print("   {}".format(self.props[item]))
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
            new_path = os.path.join(self.root,self.prop_ext,item)
            if not os.path.exists(new_path):
                os.mkdir(new_path)

        new_path = os.path.abspath(new_path)

        # Write the new path name
        with open(os.path.join(self.root,self.proppath_ext),'a') as file:
            file.write('{} {}\n'.format(item,new_path))

        self.props[item] = new_path

def setupsimim():
    """Creates and remembers the paths needed for dealing with data"""

    path = _SimIMPaths()

    print("Please specify a path to save data directories.")
    print("Specifying no path will set the path to your home directory.")
    root = input("Path: ")
    if root == '':
        root = '~'
    elif not os.path.exists(root):
        raise NameError("Specified path does not exist. Please create path and try again.")
    path._setuppath(root=root,confirm_with_user=True)