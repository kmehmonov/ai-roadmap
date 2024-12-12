import sys
import time
from datetime import date, datetime

import requests
from bs4 import BeautifulSoup


def get_soup(url: str, retry: int = 0):
    try:
        res = requests.get(url)
        with open('response.html', 'wb') as f:
            f.write(res.content)
            
        return BeautifulSoup(res.content, 'html.parser')
    except Exception:
        time.sleep(5)
        retry += 1
        print('Retrying the url:', url)
        if retry > 5:
            raise
        return get_soup(url, retry)


months = {
    'января': 1,
    'февраля': 2,
    'марта': 3,
    'апреля': 4,
    'мая': 5,
    'июня': 6,
    'июля': 7,
    'августа': 8,
    'сентября': 9,
    'октября': 10,
    'ноября': 11,
    'декабря': 12
}


def _str_to_date(date_str: str):
    """
        Converts date string into date or datetime object.
        date_str example1: 15 февраля 2024 г.
        date_str example2: Сегодня в 09:16
    """

    if 'Сегодня' in date_str:
        hour = int(date_str[-5:].split(':')[0])
        minute = int(date_str[-5:].split(':')[1])
        today = datetime.today()
        today = today.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return today
    else:
        dtuple = date_str.split()
        day = int(dtuple[0])
        month = months[dtuple[1]]
        year = int(dtuple[2])
        return date(year=year, month=month, day=day)


def split_loc_and_date(loc_and_date: str):
    dash_index = loc_and_date.rindex('-')
    loc = loc_and_date[:dash_index].strip()
    dt_str = loc_and_date[dash_index+1:].strip()
    dt = _str_to_date(dt_str)
    return (loc, dt)
