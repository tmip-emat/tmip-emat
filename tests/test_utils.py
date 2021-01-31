import pytest
from emat.util.seq_grouping import seq_int_grouper, seq_int_group_expander

def test_seq_int_grouper():
    data = [1,2,3,4,6,7,9,11,12,13,14,16,17,18,19,20]
    seq = seq_int_grouper(data)
    print(seq)
    assert seq == "1-4,6-7,9,11-14,16-20"
    data1 = seq_int_group_expander(seq)
    print(data1)
    assert data1 == data

