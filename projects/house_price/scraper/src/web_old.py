from typing import Optional
from dataclasses import dataclass
from datetime import date, timedelta

from pandas import DataFrame
from bs4 import Tag

from .functions import get_soup, split_loc_and_date


class PropertyNotFoundException(Exception):
    pass


@dataclass
class Choice:
    floor: int   # 1 to 20
    rooms: int   # 1 to 5
    is_furnished: str = 'yes'   # yes or no
    currency: str = 'UYE'   # UYE or UZS
    page: int = 1

    @property
    def url(self) -> str:
        url_template = 'https://www.olx.uz/nedvizhimost/kvartiry/arenda-dolgosrochnaya/tashkent/?currency={currency}&page={page}&search%5Bfilter_enum_furnished%5D%5B0%5D={is_furnished}&search%5Bfilter_float_floor%3Afrom%5D={floor}&search%5Bfilter_float_floor%3Ato%5D={floor}&search%5Bfilter_float_number_of_rooms%3Afrom%5D={rooms}&search%5Bfilter_float_number_of_rooms%3Ato%5D={rooms}&search%5Border%5D=created_at%3Adesc'
        url = url_template.format(
            floor=self.floor,
            rooms=self.rooms,
            is_furnished=self.is_furnished,
            currency=self.currency,
            page=self.page,
        )
        return url


class Price:
    """
    price_str example 1: 53 у.е.Договорная
    price_str example 2: 400 у.е.
    """

    def __init__(self, price_str: str):
        index = price_str.rindex(' ')
        self.amount = price_str[:index].strip().replace(' ', '')

        info = price_str[index+1:].strip()
        if (i := info.find('Договорная')) != -1:
            self.is_negotiable = 'yes'
            self.currency = info[:i]
        else:
            self.is_negotiable = 'no'
            self.currency = info


class Card:
    selectors = {
        # 'title': 'h6.css-16v5mdi',
        'title': 'h4.css-1s3qyje',
        # 'price': 'p.css-10b0gli',
        'price': 'p.css-13afqrm',
        # 'loc_and_date': 'p.css-1a4brun',
        'loc_and_date': 'p.css-1mwdrlh',
        # 'area': 'span.css-643j0o',
        'area': 'span.css-1cd0guq',
    }


    columns = (
        'title',
        'price',
        'currency',
        'is_negotiable',
        'location',
        'date',
        'area',
        'floor',
        'rooms'
    )

    def __init__(self, tag: Tag, choice: Choice):
        self.tag = tag        
        self.title = self._select('title')
        price_str = self._select('price')
        self.price = Price(price_str)
        loc_and_date = self._select('loc_and_date')
        self.location, self.date = split_loc_and_date(loc_and_date)
        self.area = self._select('area')
        self.floor = choice.floor
        self.rooms = choice.rooms

    def _select(self, key):
        prop = self.tag.select_one(self.selectors[key])
        if prop is None:
            raise PropertyNotFoundException(
                prop, 'not found for selector', self.selectors[key]
            )
        return prop.text.strip().replace('\n', '')

    @property
    def data(self):
        return (
            self.title,
            self.price.amount,
            self.price.currency,
            self.price.is_negotiable,
            self.location,
            str(self.date),
            self.area,
            self.floor,
            self.rooms
        )

    def is_posted_yesterday(self):
        yesterday = date.today() - timedelta(days=1)
        return self.date == yesterday


class Page:
    all_tags_selector = {'data-cy': 'l-card'}
    error_selector = 'p.css-1oc165u'

    def __init__(self, choice: Choice):
        self.choice = choice
        self._cards: Optional[list[Card]] = None

        soup = get_soup(self.choice.url)
        paginator = soup.select('.pagination-item')
        if paginator:
            self.last_page = int(paginator[-1].text)
        else:
            self.last_page = 1

    @property
    def cards(self) -> list[Card]:
        if self._cards:
            return self._cards

        cards = []
        for page in range(1, self.last_page + 1):
            self.choice.page = page

            print(
                'Downloading page', page
            )   # NOTE: for debugging
            soup = get_soup(self.choice.url)

            error = soup.select_one(self.error_selector)
            if error:
                print('Page is blank')
                continue

            all_tags = soup.find_all('div', self.all_tags_selector)

            for tag in all_tags:
                card = Card(tag, self.choice)
                if card.is_posted_yesterday():
                    cards.append(card.data)

        self._cards = cards
        return self._cards

    @property
    def as_df(self) -> DataFrame:
        df = DataFrame(data=self.cards, columns=Card.columns)
        df = df.drop_duplicates()
        return df
