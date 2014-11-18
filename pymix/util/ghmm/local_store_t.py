from pymix.util.ghmm.types import kHigherOrderEmissions
from pymix.util.ghmm.wrapper import ARRAY_CALLOC, ARRAY_MALLOC


class local_store_t():
    def __init__(self):
        self.pi_num = None
        self.pi_denom = 0.0
        self.a_num = None
        self.a_denom = None
        self.b_num = None
        self.b_denom = None



    def reestimate_init(self, mo):
        for i in range(mo.N):
            self.pi_num[i] = 0.0
            self.a_denom[i] = 0.0
            for j in range(mo.N):
                self.a_num[i][j] = 0.0

            if mo.model_type & kHigherOrderEmissions:
                size = pow(mo.M, mo.order[i])
                for m in range(size):
                    self.b_denom[i][m] = 0.0
                size *= mo.M
                for m in range(size):
                    self.b_num[i][m] = 0.0
            else:
                self.b_denom[i][0] = 0.0
                for m in range(mo.M):
                    self.b_num[i][m] = 0.0
        self.pi_denom = 0.0
        return self

def reestimate_alloc(mo):
    self = local_store_t()
    self.pi_num = ARRAY_CALLOC(mo.N)
    self.a_num = ARRAY_CALLOC(mo.N)
    for i in range(mo.N):
        self.a_num[i] = ARRAY_CALLOC(mo.N)
    self.a_denom = ARRAY_CALLOC(mo.N)

    self.b_num = ARRAY_MALLOC(mo.N)
    self.b_denom = ARRAY_MALLOC(mo.N)

    if mo.model_type & kHigherOrderEmissions:
        for i in range(mo.N):
            self.b_num[i] = ARRAY_CALLOC(pow(mo.M, mo.order[i] + 1))
            self.b_denom[i] = ARRAY_CALLOC(pow(mo.M, mo.order[i]))

    else:
        for i in range(mo.N):
            self.b_num[i] = ARRAY_CALLOC(mo.M)
            self.b_denom[i] = ARRAY_CALLOC(1)
    return self
