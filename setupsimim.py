import sys, os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

from simim._paths import create_paths

create_paths()