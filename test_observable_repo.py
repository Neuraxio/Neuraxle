
from abc import ABC, abstractmethod
import neuraxle.metaopt.auto_ml
#hyperparamsjson will inhirt from the observable (create new package for observable class) 
class Observable(metaclass=abc.ABCMeta):
    def __init__(self):
        self.obs = []
        self.changed = 0
        Synchronization.__init__(self)

    def subscribe(self, observer):
        if observer not in self.obs:
            self.obs.append(observer)

    def deleteObserver(self, observer):
        self.obs.remove(observer)

    def notifyObservers(self, arg = None):
        '''If 'changed' indicates that this object
        has changed, notify all its observers, then
        call clearChanged(). Each observer has its
        update() called with two arguments: this
        observable object and the generic 'arg'.'''

        self.mutex.acquire()
        try:
            if not self.changed: return
            # Make a local copy in case of synchronous
            # additions of observers:
            localArray = self.obs[:]
            self.clearChanged()
        finally:
            self.mutex.release()
        # Updating is not required to be synchronized:
        for observer in localArray:
            observer.update(self, arg)




class Observer(metaclass=abc.ABCMeta):
    def __init__(self):
        self._subject = None
        self._observer_state = None
    @abc.abstractmethod
    def update(self, arg):
        pass
def test_repository_should_notify(tmpdir):
      observer = Observer()
      repo = HyperparamsJsonRepository(cache_folder=tmpdir)
      repo.subscribe(observer)


      repo.save_trial(Trial())

assert observer.received_trial()



#f=auto_ml.HyperparamsJsonRepository(Observable)
#create instance for observer class :
#obj=Observer()
#f.subscribe(obj)
#f.notify_observer()

#or
test_repository_should_notify(Observable)
