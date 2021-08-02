import uuid
import datetime


def uuid1_time(u):
    return datetime.datetime.fromtimestamp((u.time - 0x01b21dd213814000)*100/1e9)

