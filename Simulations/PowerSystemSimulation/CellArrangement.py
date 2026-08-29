from abc import abstractmethod, ABC


class CellArrangement(ABC):

    @abstractmethod
    def get_pack_voltage(self, cell_voltage: float) -> float:
        pass

    @abstractmethod
    def get_cell_current(self, pack_current: float) -> float:
        pass


class MultiCellArrangement(CellArrangement, ABC):
    __inner_arrangement: CellArrangement
    __number: int  # how often the internal arrangement is repeated

    def __init__(self, inner_arrangement: CellArrangement, number: int):
        if number <= 1:
            raise ValueError("Unexpected Cell number. Must be greater than 1.")
        self.inner_arrangement = inner_arrangement
        self.number = number


class SeriesCellArrangement(MultiCellArrangement):

    def get_pack_voltage(self, cell_voltage: float):
        return self.inner_arrangement.get_pack_voltage(cell_voltage * self.number)

    def get_cell_current(self, pack_current: float):
        return self.inner_arrangement.get_cell_current(pack_current)


class ParallelCellArrangement(MultiCellArrangement):

    def get_pack_voltage(self, cell_voltage: float):
        return self.inner_arrangement.get_pack_voltage(cell_voltage)

    def get_cell_current(self, pack_current: float):
        return self.inner_arrangement.get_cell_current(pack_current) / self.number


class SingleCell(CellArrangement):

    def get_pack_voltage(self, cell_voltage: float):
        return cell_voltage

    def get_cell_current(self, pack_current: float):
        return pack_current
