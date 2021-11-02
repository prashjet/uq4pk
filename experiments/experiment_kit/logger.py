import sys, os

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, 'w+')  # the "+" means that if the logfile does not exist, it is created.
        self.orig_out = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

    def activate(self):
        sys.stdout = self

    def deactivate(self):
        sys.stdout = self.orig_out