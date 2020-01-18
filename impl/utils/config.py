import datetime

progress = True
messages = False


def debug(msg):
    if messages:
        print(f'[{datetime.datetime.now()}] {msg}')
