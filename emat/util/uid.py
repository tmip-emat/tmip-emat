import uuid
import datetime


def uuid_time(u):
    uh = u.hex
    if uh[12] == '6':
        t = int('0x'+uh[:12]+uh[13:16], 16)
    elif uh[12] == '4':
        raise ValueError("time not available from uuid version 4")
    else:
        t = u.time
    return datetime.datetime.fromtimestamp((t - 0x01b21dd213814000)*100/1e9)


def uuid6(uuid1=None, node=None, clock_seq=None):
    """
    Create a (draft) UUID version 6 per draft spec.

    http://gh.peabody.io/uuidv6/

    Parameters
    ----------
    uuid1
    node
    clock_seq

    Returns
    -------
    UUID
    """
    if isinstance(uuid1, bytes):
        uuid1 = uuid.UUID(bytes=uuid1)
    if uuid1 is None:
        uuid1 = uuid.uuid1(node, clock_seq)
    uh = uuid1.hex
    if uh[12] == '6':
        # already a version 6 UUID, return it
        return uuid1
    elif uh[12] != '1':
        import warnings
        warnings.warn(f"cannot convert UUID v{uh[12]} to v6")
        return uuid1
    tlo1 = uh[:5]
    tlo2 = uh[5:8]
    tmid = uh[8:12]
    thig = uh[13:16]
    rest = uh[16:]
    uh6 = thig + tmid + tlo1 + '6' + tlo2 + rest
    return uuid.UUID(hex=uh6)
