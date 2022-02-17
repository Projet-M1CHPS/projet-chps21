import os
import sys
from os.path import exists
from os import fork

# == neural-network interface ==

import kreps

# == ======================== ==



def onPrecisionChanged(value):
	print(f"onPrecisionChanged: {value}")


def start(cli_args):
	if len(cli_args) == 0:
		raise Exception(f"No configuration file given. \nStop.")
	elif not exists(cli_args[0]):
		raise Exception(f"Configuration file [{cli_args[0]}] does not exists at this location. \nStop.")

	c = kreps.NetworkInterface(cli_args[0])
	print("InterfaceObject version is: %s" % kreps.getVersion())
	print("Made a networkInterface called !")
	c.printJSONConfig()
	c.onPrecisionChanged(onPrecisionChanged)

	run_pid = os.fork()
	if run_pid < 0:  # error
		raise Exception("Error on fork")
	if run_pid == 0:  # child
		c.createAndTrain()
		exit(0)
	print(f"runner: pid of child is: {run_pid}")
	return run_pid


# training_thread.join()


if __name__ == '__main__':
	pid = start(sys.argv[1:])
	os.waitpid(pid, 0)
