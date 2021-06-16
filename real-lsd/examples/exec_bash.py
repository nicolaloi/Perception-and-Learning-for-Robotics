import os
import real_lsd
import subprocess

path     = os.path.dirname(real_lsd.__file__)
abs_path = path + '/envs/settings/landing/cpptest.json'
cp_path  = '/media/scratch1/nasib/data/'

list_files = subprocess.run(["cp", abs_path, cp_path])
print("The exit code was: %d" % list_files.returncode)
