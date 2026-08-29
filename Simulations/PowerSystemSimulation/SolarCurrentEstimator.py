

class SolarCurrentEstimator(ABC):
    @abstractmethod
    def get_current(self, solar_power: float) -> float:
        pass

class ConstantCurrentEstimator(SolarCurrentEstimator):
    def __init__(self, current: float, ):
        self.current = current

    def get_current(self, solar_power: float) -> float:
        return self.current