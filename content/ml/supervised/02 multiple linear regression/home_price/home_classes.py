import pandas as pd

class Home:
    """Represents a single home with validated fields and encapsulation."""
    ALLOWED_STATUSES = {"new", "old"}
    ALLOWED_FURNISHING = {"furnished", "unfurnished"}
    ALLOWED_BATHROOM = {"shared", "private"}

    def __init__(self, price, area, living_area, kitchen_area, number_of_rooms,
                 status, furnishing_status, bathroom, floor, total_floors,
                 built_year, address, with_makler, home_ad_link=None):
        self.set_price(price)
        self.set_area(area)
        self.set_living_area(living_area)
        self.set_kitchen_area(kitchen_area)
        self.set_number_of_rooms(number_of_rooms)
        self.set_status(status)
        self.set_furnishing_status(furnishing_status)
        self.set_bathroom(bathroom)
        self.set_floor(floor)
        self.set_total_floors(total_floors)
        self.set_built_year(built_year)
        self.set_address(address)
        self.set_with_makler(with_makler)
        self.set_home_ad_link(home_ad_link)

    # Numeric fields with None support
    def set_price(self, value):
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError("price must be a positive number or None.")
        self._price = float(value) if value is not None else None

    def set_area(self, value):
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError("area must be a positive number or None.")
        self._area = float(value) if value is not None else None

    def set_living_area(self, value):
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError("living_area must be a positive number or None.")
        self._living_area = float(value) if value is not None else None

    def set_kitchen_area(self, value):
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError("kitchen_area must be a positive number or None.")
        self._kitchen_area = float(value) if value is not None else None

    def set_number_of_rooms(self, value):
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError("number_of_rooms must be an integer ≥ 1 or None.")
        self._number_of_rooms = value

    def set_floor(self, value):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError("floor must be a non-negative integer or None.")
        self._floor = value

    def set_total_floors(self, value):
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError("total_floors must be a positive integer or None.")
        self._total_floors = value

    def set_built_year(self, value):
        if value is not None and (not isinstance(value, int) or value < 1800 or value > 2100):
            raise ValueError("built_year must be an integer between 1800 and 2100, or None.")
        self._built_year = value

    def set_address(self, value):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("address must be a non-empty string or None.")
        self._address = value.strip() if value is not None else None

    def set_with_makler(self, value):
        if value is not None and not isinstance(value, bool):
            raise ValueError("with_makler must be a boolean or None.")
        self._with_makler = value

    # Categorical fields with None support
    def set_status(self, value):
        if value is not None and value not in self.ALLOWED_STATUSES:
            raise ValueError(f"status must be one of {self.ALLOWED_STATUSES} or None.")
        self._status = value

    def set_furnishing_status(self, value):
        if value is not None and value not in self.ALLOWED_FURNISHING:
            raise ValueError(f"furnishing_status must be one of {self.ALLOWED_FURNISHING} or None.")
        self._furnishing_status = value

    def set_bathroom(self, value):
        if value is not None and value not in self.ALLOWED_BATHROOM:
            raise ValueError(f"bathroom must be one of {self.ALLOWED_BATHROOM} or None.")
        self._bathroom = value
    
    def set_home_ad_link(self, value):
        if value is not None and (not isinstance(value, str) or not value.startswith("http")):
            raise ValueError("home_ad_link must be a valid URL or None.")
        self._home_ad_link = value

    # Convert to dictionary
    def to_dict(self, home_id):
        return {
            "home_id": home_id,
            "price": self._price,
            "area": self._area,
            "living_area": self._living_area,
            "kitchen_area": self._kitchen_area,
            "number_of_rooms": self._number_of_rooms,
            "status": self._status,
            "furnishing_status": self._furnishing_status,
            "bathroom": self._bathroom,
            "floor": self._floor,
            "total_floors": self._total_floors,
            "built_year": self._built_year,
            "address": self._address,
            "with_makler": self._with_makler,
            "home_ad_link": self._home_ad_link
        }
    
    def __repr__(self):
        return (f"Home(price={self._price}, area={self._area}, living_area={self._living_area}, "
            f"kitchen_area={self._kitchen_area}, number_of_rooms={self._number_of_rooms}, "
            f"status={self._status}, furnishing_status={self._furnishing_status}, "
            f"bathroom={self._bathroom}, floor={self._floor}, total_floors={self._total_floors}, "
            f"built_year={self._built_year}, address={repr(self._address)}, with_makler={self._with_makler}, "
            f"home_ad_link={repr(self._home_ad_link)})")




class HomeCollection:
    """Manages a collection of Home instances with auto-incrementing home_id."""
    def __init__(self):
        self._homes = {}
        self._next_id = 1

    def add_home(self, home):
        """Add a new Home to the collection."""
        if not isinstance(home, Home):
            raise ValueError("Only Home instances can be added.")
        self._homes[self._next_id] = home
        self._next_id += 1

    def delete_home_by_id(self, home_id):
        """Delete a Home by its ID."""
        if home_id in self._homes:
            del self._homes[home_id]
        else:
            raise KeyError(f"No home found with ID: {home_id}")

    def get_home_by_id(self, home_id):
        """Retrieve a Home by ID."""
        return self._homes.get(home_id, None)

    def get_all_homes(self):
        """Return all homes as a pandas DataFrame."""
        data = [home.to_dict(hid) for hid, home in self._homes.items()]
        return pd.DataFrame(data)

    def get_all_ids(self):
        return list(self._homes.keys())

    def __repr__(self):
        return f"HomeCollection({len(self._homes)} homes)"