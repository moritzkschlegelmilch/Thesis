from abc import abstractmethod, ABC


class Scorable(ABC):

    def __init__(self, eps):
        self.eps = eps

    @abstractmethod
    def prepare(self, ocel):
        pass

    @abstractmethod
    def assign_score_push(self, o1, o2) -> float:
        pass

    @abstractmethod
    def assign_score_pull(self, o1, o2) -> float:
        pass
