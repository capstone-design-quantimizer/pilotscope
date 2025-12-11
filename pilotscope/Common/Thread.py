from threading import Thread

from pilotscope.Common.Util import pilotscope_exit


class ValueThread(Thread):

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=True):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.result = None
        self.exception = None  # Store exception if any

    def run(self):
        if self._target is not None:
            try:
                self.result = self._target(*self._args, **self._kwargs)
            except Exception as e:
                print(f"\n❌ Exception in thread {self.name}:")
                print(f"   {str(e)}")
                self.exception = e  # Store the exception for later retrieval
                import traceback
                traceback.print_exc()
                raise e

    def join(self, timeout=None):
        super().join(timeout=timeout)
        # If an exception occurred, re-raise it in the calling thread
        if self.exception:
            raise self.exception
        return self.result
