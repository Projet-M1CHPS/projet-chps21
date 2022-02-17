import os
import random

import signal
import time

duel_start_at = time.time() + 3
global parent_pid


def safe_exit():
	if os.getpid() != parent_pid:
		exit(0)
	else:
		raise Exception("End of the duel")


class Duelist:
	adversary_pid: int = 0

	def loose(self):
		os.kill(self.adversary_pid, signal.Signals.SIGUSR2)  # win signal

	def shoot(self):
		time.sleep(random.random())
		print(f"[{os.getpid()}]: PAN !")
		os.kill(self.adversary_pid, signal.Signals.SIGUSR1)  # shot signal

	def _duel(self):
		print(f"[{os.getpid()}]: Ready to fire...")
		time.sleep(1)
		self.shoot()

	def set_duelist_adversary(self, adversary_pid: int) -> None:
		self.adversary_pid = adversary_pid
		trap_duelist_signals()
		random.seed(os.getpid())
		print(f"[{os.getpid()}]: In place.")
		wait_time = duel_start_at - time.time()
		time.sleep(wait_time if wait_time > 0 else 1)
		self._duel()


global me


def duelist_sig_handler(signum, stack):
	if signum == signal.Signals.SIGUSR1:
		print(f"[{os.getpid()}]: Argh...")
		if random.random() < .33:
			print(f"[{os.getpid()}]: Well tried cowboy.")
			me.shoot()
			return
		print(f"[{os.getpid()}]: You killed me.")
		me.loose()
		safe_exit()
	elif signum == signal.Signals.SIGUSR2:
		print(f"[{os.getpid()}]: I win.")
		safe_exit()


def trap_duelist_signals():
	signal.signal(signal.Signals.SIGUSR1, duelist_sig_handler)
	signal.signal(signal.Signals.SIGUSR2, duelist_sig_handler)


def _create_duelist() -> int:
	duelist_pid = os.fork()
	if duelist_pid < 0:  # error
		raise Exception("Error on fork")
	if duelist_pid == 0:  # child
		global me
		me = Duelist()
		me.set_duelist_adversary(os.getppid())
		while 1:
			time.sleep(1e-3)
	return duelist_pid


def start():
	global parent_pid
	parent_pid = os.getpid()
	global me
	me = Duelist()
	me.set_duelist_adversary(_create_duelist())
	while 1:
		time.sleep(1e-3)
