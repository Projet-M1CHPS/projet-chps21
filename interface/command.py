import time
from cmd import Cmd

import GUI
import main as prog

import duel as duel

import sys
import signal
from os import waitpid, kill

import GUI.GUI

global command_obj


# region Colored output


def toRed(skk): return "\033[91m {}\033[00m".format(skk)


def toGreen(skk): return "\033[92m {}\033[00m".format(skk)


def toYellow(skk): return "\033[93m {}\033[00m".format(skk)


def toLightPurple(skk): return "\033[94m {}\033[00m".format(skk)


def toMagenta(skk): return "\033[35m {}\033[00m".format(skk)


def toPurple(skk): return "\033[95m {}\033[00m".format(skk)


def toCyan(skk): return "\033[96m {}\033[00m".format(skk)


def toLightGray(skk): return "\033[97m {}\033[00m".format(skk)


def toBlack(skk): return "\033[98m {}\033[00m".format(skk)


# endregion

# region Signal handling

SIGNALS_TO_NAMES_DICT = dict((getattr(signal, n), n) for n in dir(signal) if n.startswith('SIG') and '_' not in n)
print(SIGNALS_TO_NAMES_DICT)


def sigHandler(signum, stack):
	if signum in [1, 2, 3, 15]:
		if command_obj.is_subprocess_running() and not command_obj.is_waiting_for_ask():
			command_obj.pause_subprocess()
			choice = command_obj.ask("Is process is running, do you want to force stop [y/n] ? ")
			if choice.lower().strip() != "y":
				command_obj.resume_subprocess()
				return
			else:
				command_obj.stop_subprocess()
				return
		print('Caught signal %s (%s), exiting.' % (SIGNALS_TO_NAMES_DICT[signum], str(signum)))
		command_obj.do_exit()
		sys.exit()


def initSignalTrap():
	for sig in signal.Signals:
		try:
			signal.signal(sig, sigHandler)
		except OSError:
			pass


# print('Skipping', sig)


# endregion


class MyPrompt(Cmd):
	subprocess_pid: int = 0
	asking: bool = False
	cli_args = []
	prompt = toMagenta("kreps$ ")
	intro = toPurple("---------- KREPS ----------")

	def setCliArguments(self, cli_args):
		self.cli_args = cli_args

	def parseAndExecuteArguments(self):
		for each in self.cli_args:
			print(each)
			self.onecmd(each.lower().strip().lstrip('-'))

	def preloop(self) -> None:
		Cmd.do_help(self, "")
		self.parseAndExecuteArguments()

	@staticmethod
	def my_prompt(msg="", line_trail=True):
		if line_trail:
			print(msg)
		else:
			print(msg, end='')
			sys.stdout.flush()

	def is_waiting_for_ask(self) -> bool:
		return self.asking

	def ask(self, msg="") -> str:
		self.asking = True
		sys.stdout.flush()
		self.my_prompt(msg, False)
		# choice = sys.stdin.read(1)
		choice = input()
		# print(choice, end='')
		sys.stdout.flush()
		self.asking = False
		return choice

	def do_exit(self, inp=""):
		self.stop_subprocess()
		self.my_prompt("Bye")
		return True

	def help_exit(self):
		self.my_prompt('exit the application. Shorthands: q  Ctrl-D.')

	def do_gui(self, inp=""):
		self.my_prompt("Starting the GUI.")
		GUI.GUI.start()
		self.my_prompt("Closing the GUI.")

	def do_duel(self, inp=""):
		try:
			duel.start()
		except Exception as e:
			print(e)

	def do_run(self, inp=""):
		try:
			self.subprocess_pid = prog.start(inp.split(' '))
			print(f"command: pid of child is: {self.subprocess_pid}")
			waitpid(self.subprocess_pid, 0)
		except Exception as e:
			print(e, file=sys.stderr)
		finally:
			self.subprocess_pid = 0

	def help_gui(self):
		self.my_prompt('Launch the GUI interface')

	def default(self, inp=""):
		self.my_prompt(inp)
		if inp.strip() == 'q':
			return self.do_exit(inp)

		self.my_prompt("Unknown command: {}".format(inp))

	do_EOF = do_exit
	help_EOF = help_exit

	def is_subprocess_running(self):
		return self.subprocess_pid > 0

	def pause_subprocess(self):
		if not self.is_subprocess_running():
			return
		self.my_prompt("Pausing the current subprocess")
		kill(self.subprocess_pid, 19)  # SIGSTOP

	def resume_subprocess(self):
		if not self.is_subprocess_running():
			return
		self.my_prompt("Resuming the current subprocess")
		kill(self.subprocess_pid, 18)  # SIGCONT

	def stop_subprocess(self):
		if not self.is_subprocess_running():
			return
		self.resume_subprocess()
		self.my_prompt("Killin' da subprocess like 2pac")
		kill(self.subprocess_pid, 6)  # SIGABRT


if __name__ == '__main__':
	command_obj = MyPrompt()
	command_obj.setCliArguments(sys.argv[1:] if len(sys.argv) > 1 else [])
	initSignalTrap()

	command_obj.cmdloop()
