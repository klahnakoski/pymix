#
#   A C-program for MT19937, with initialization improved 2002/1/26.
#   Coded by Takuji Nishimura and Makoto Matsumoto.
#
#   Before using, initialize the state by using init_genrand(seed)
#   or init_by_array(init_key, key_length).
#
#   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
#   All rights reserved.
#   Copyright (C) 2005, Mutsuo Saito,
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions
#   are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. The names of its contributors may not be used to endorse or promote
#        products derived from this software without specific prior written
#        permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR
#   PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
#   Any feedback is very welcome.
#   http:# www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
#   email: m-mat @ math.sci.hiroshima-u.ac.jp (space)
#

# Period parameters
N = 624
M = 397
mag01 = [0x0, 0x9908b0df]  # mag01[x] = x * MATRIX_A  for x=0,1

UPPER_MASK = 0x80000000  # most significant w-r bits
LOWER_MASK = 0x7fffffff  # least significant r bits



class Random(object):

    @staticmethod
    def set_seed(seed):
        SEED.init_genrand(seed)

    @staticmethod
    def float():
        return SEED.genrand_real2()

    @staticmethod
    def int32():
        return SEED.genrand_int32()


    def __init__(self):
        self.mt = [0] * N             # the array for the state vector
        self.mti = N + 1              # mti==N+1 means mt[N] is not initialized

    # initializes mt[N] with a seed
    def init_genrand(self, s):
        self.mt[0] = s & 0xffffffff
        for i in range(1, N):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i)
            # See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier.
            # In the previous versions, MSBs of the seed affect
            # only MSBs of the array mt[].
            # 2002/01/09 modified by Makoto Matsumoto
            self.mt[i] &= 0xffffffff  # for >32 bit machines
        self.mti = N

    # initialize by an array with array-length
    # init_key is the array for initializing keys
    # key_length is its length
    # slight change for C++, 2004/2/26
    def init_by_array(self, init_key, key_length):
        self.init_genrand(19650218)
        i = 1
        j = 0
        for k in range(max(key_length, N)):
            self.mt[i] = (self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1664525)) + init_key[j] + j  # non linear
            self.mt[i] &= 0xffffffff  # for WORDSIZE > 32 machines
            i += 1
            j += 1
            if i >= N:
                self.mt[0] = self.mt[N - 1]
                i = 1
            if j >= key_length:
                j = 0

        for k in range(N - 1):
            self.mt[i] = (self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1566083941)) - i  # non linear
            self.mt[i] &= 0xffffffff  # for WORDSIZE > 32 machines
            i += 1
            if i >= N:
                self.mt[0] = self.mt[N - 1]
                i = 1

        self.mt[0] = 0x80000000 # MSB is 1 assuring non-zero initial array


    # generates a random number on [0,0xffffffff]-interval
    def genrand_int32(self):

        if self.mti >= N: # generate N words at one time
            if self.mti == N + 1:   # if init_genrand(has not been called,
                self.init_genrand(5489) # a default initial seed is used

            for kk in range(N - M):
                y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK)
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1]

            for kk in range(N - M, N - 1):
                y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK)
                self.mt[kk] = self.mt[kk - (N - M)] ^ (y >> 1) ^ mag01[y & 0x1]

            y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK)
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1]

            self.mti = 0

        y = self.mt[self.mti]
        self.mti += 1

        # Tempering
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)

        # # CONVERT TO TWO'S COMPLEMENT
        # if y & 0x80000000:
        #     return y - 0x100000000
        return y


    # generates a random number on [0,0x7fffffff]-interval
    def genrand_int31(self):
        return self.genrand_int32() >> 1


    # generates a random number on [0,1]-real-interval
    def genrand_real1(self):
        return self.genrand_int32() * (1.0 / 4294967295.0)    # divided by 2^32-1


    # generates a random number on [0,1)-real-interval
    def genrand_real2(self):
        return self.genrand_int32() / 4294967296.0    # divided by 2^32


    # generates a random number on (0,1)-real-interval
    def genrand_real3(self):
        return (self.genrand_int32() + 0.5) * (1.0 / 4294967296.0)    # divided by 2^32


    # generates a random number on [0,1) with 53-bit resolution
    def genrand_res53(self):
        a = self.genrand_int32() >> 5
        b = self.genrand_int32() >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

        # These real versions are due to Isaku Wada, 2002/01/09 added


SEED = Random()
