from src.db import DBWriter
from scraper.src.web_old import Choice, Page


def download_for_floor(floor: int, is_furnished: str):
    for rooms in range(1, 6):
        try:
            choice = Choice(
                floor=floor, rooms=rooms, is_furnished=is_furnished
            )
            page = Page(choice)
            print('Floor:', choice.floor, ', Rooms:', choice.rooms, end=' ')
            print('There are', page.last_page, 'pages.')
            df = page.as_df
            df['is_furnished'] = is_furnished
            
            writer = DBWriter(df=df, format='sqlite')
            writer.save()

        except Exception:
            print('********ERROR********')
            print('Floor:', choice.floor, ', Rooms:', choice.rooms)
            raise



def main():
    print('SMebel:\n')
    for floor in range(1, 21):
        download_for_floor(floor=floor, is_furnished='yes')

    print('BezMebel:\n')
    for floor in range(1, 21):
        download_for_floor(floor=floor, is_furnished='no')

if __name__ == '__main__':
    print("""Houses can be located from the first to the twentieth floor.
Each apartment has from one to five rooms.
Houses can be furnished (SMebel) or unfurnished (BezMebel).
""")
    main()