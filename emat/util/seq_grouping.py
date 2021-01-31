from itertools import groupby
from operator import itemgetter


def seq_int_grouper(data):
    groupings = []
    for k, g in groupby(enumerate(data), (lambda ix : ix[0] - ix[1])):
        agg = list(map(itemgetter(1), g))
        if len(agg)==1:
            groupings.append(str(agg[0]))
        else:
            groupings.append(f"{agg[0]}-{agg[-1]}")
    return ",".join(groupings)

def seq_int_group_expander(seq):
    seq = seq.split(",")
    data = []
    for agg in seq:
        if "-" in agg:
            first, last = agg.split("-")
            data.extend(range(int(first), int(last)+1))
        else:
            data.append(int(agg))
    return data

