import pickle
import sys
import pytest
import timeit
from ding.utils import shm_encode_with_schema, shm_decode, equal
from dizoo.distar.envs import fake_rl_data_batch_with_last


@pytest.mark.lxl
def test_shm_encode_decode():
    data = fake_rl_data_batch_with_last(unroll_len=64)
    encoding, schema = shm_encode_with_schema(data)
    decoding = shm_decode(encoding, schema)
    assert equal(data, decoding)


class ShmSpeedTest:

    def __init__(self) -> None:
        self.data = fake_rl_data_batch_with_last(unroll_len=64)
        # self.data64 = torch.rand(1024, 1024, 16)
        self.encoding = None
        self.schema = None
        self.decoding = None

    def encode(self):
        self.encoding, self.schema = shm_encode_with_schema(self.data)

    def decode(self):
        self.decoding = shm_decode(self.encoding, self.schema)

    def pickle_encode(self):
        # print(pickle.HIGHEST_PROTOCOL)
        self.p_encoding = pickle.dumps(self.data)

    def pickle_load(self):
        self.p_decoding = pickle.loads(self.p_encoding)


@pytest.mark.lxl
def test_shm_encode_decode_speed():
    test = ShmSpeedTest()

    print(timeit.repeat(test.encode, repeat=3, number=10))
    print(timeit.repeat(test.decode, repeat=3, number=10))

    # print(timeit.repeat(test.pickle_encode, repeat=3, number=10))
    # print(timeit.repeat(test.pickle_load, repeat=3, number=10))
