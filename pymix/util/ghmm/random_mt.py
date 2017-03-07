#
#   A C-program for MT19937, with initialization improved 2002/1/26.
#   Coded by Takuji Nishimura and Makoto Matsumoto.

# Period parameters


BATCH_SIZE = 624
HALF = 397
mag01 = [0x0, 0x9908b0df]  # mag01[x] = x * MATRIX_A  for x=0,1

UPPER_MASK = 0x80000000  # most significant w-r bits
LOWER_MASK = 0x7fffffff  # least significant r bits
LIMIT_32BITS = 0xffffffff


class Random(object):
    def __init__(self, seed=5489):
        self.batch = [0] * BATCH_SIZE
        self._reset(seed)

    def _reset(self, seed):
        self.batch[0] = seed & LIMIT_32BITS
        for i in range(1, BATCH_SIZE):
            self.batch[i] = (1812433253 * (self.batch[i - 1] ^ (self.batch[i - 1] >> 30)) + i)
            self.batch[i] &= LIMIT_32BITS
        self.next = BATCH_SIZE

    def _generate_batch(self):
        first_half = BATCH_SIZE - HALF
        for kk in range(first_half):
            y = (self.batch[kk] & UPPER_MASK) | (self.batch[kk + 1] & LOWER_MASK)
            self.batch[kk] = self.batch[kk + HALF] ^ (y >> 1) ^ mag01[y & 0x1]
        for kk in range(first_half, BATCH_SIZE - 1):
            y = (self.batch[kk] & UPPER_MASK) | (self.batch[kk + 1] & LOWER_MASK)
            self.batch[kk] = self.batch[kk - first_half] ^ (y >> 1) ^ mag01[y & 0x1]
        y = (self.batch[BATCH_SIZE - 1] & UPPER_MASK) | (self.batch[0] & LOWER_MASK)
        self.batch[BATCH_SIZE - 1] = self.batch[HALF - 1] ^ (y >> 1) ^ mag01[y & 0x1]
        self.next = 0

    def _next(self):
        """
        RETURN uint32
        """
        if self.next >= BATCH_SIZE:
            self._generate_batch()

        y = self.batch[self.next]
        self.next += 1

        # TEMPER
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)

        return y

    def uint32(self):
        # # CONVERT TO TWO'S COMPLEMENT
        # if y & 0x80000000:
        #     return y - 0x100000000
        return self._next()

    # generates a random number on [0,0x7fffffff]-interval
    def uint31(self):
        return self._next() >> 1

    # generates a random number on [0,1]-real-interval
    def real1(self):
        return self._next() * (1.0 / 4294967295.0)    # divided by 2^32-1

    # generates a random number on [0,1)-real-interval
    def float23(self):
        return self._next() / 4294967296.0    # divided by 2^32

    # generates a random number on (0,1)-real-interval
    def real3(self):
        return (self._next() + 0.5) * (1.0 / 4294967296.0)    # divided by 2^32

    # generates a random number on [0,1) with 53-bit resolution
    def float53(self):
        a = self._next() >> 5
        b = self._next() >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

        # These real versions are due to Isaku Wada, 2002/01/09 added


SEED = Random()


def set_seed(seed):
    SEED._reset(seed)


def float23():
    return SEED.float23()


def float53():
    return SEED.float53()


def uint32():
    return SEED.uint32()
