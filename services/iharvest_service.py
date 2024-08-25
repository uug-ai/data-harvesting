from abc import ABC, abstractmethod


class IHarvestService(ABC):
    @abstractmethod
    def connect(self, agent):
        pass