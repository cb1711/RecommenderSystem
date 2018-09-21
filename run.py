import sys
import os
import subprocess

executable = "bin/ocular"

hosts = '' if not os.environ.get('MPI_HOSTS') else '-f {0}'.format(os.environ.get('MPI_HOSTS'))

procs = input("Enter the number of processes: ")

sys_call = '{0} -n {1} {2} ./{3}'.format("mpirun", procs , "-f hosts", executable)

print(sys_call)
subprocess.call([sys_call], shell=True)
