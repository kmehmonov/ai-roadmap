from dataclasses import dataclass




@dataclass
class Filter:
    city: str
    location: str
    floor: str
    type_of_house: str
    rooms: int
    brokerage_fee: bool
    furnished: bool
