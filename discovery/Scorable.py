from abc import abstractmethod, ABC


class Scorable(ABC):

    def __init__(self, eps):
        self.eps = eps

    @abstractmethod
    def assign_score_push(self, o1, o2, data) -> float:
        pass

    @abstractmethod
    def assign_score_pull(self, o1, o2, data) -> float:
        pass


class DataSource(ABC):

    def __init__(self, scorables: list[Scorable]):
        self.scorables = scorables

    @abstractmethod
    def prepare(self, ocel):
        pass

    def assign_score(self, fallback) -> float:
        res = 0
        for scorable in self.scorables:
            res += scorable.eps * fallback(scorable)

        return res

    def assign_score_push(self, o1, o2) -> float:
        return self.assign_score(lambda scorable: scorable.assign_score_push(o1, o2, self))

    def assign_score_pull(self, o1, o2) -> float:
        return self.assign_score(lambda scorable: scorable.assign_score_pull(o1, o2, self))

    def weight(self):
        res = 0
        for scoreable in self.scorables:
            res += scoreable.eps

        return res
