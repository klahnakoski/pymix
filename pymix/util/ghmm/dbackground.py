from pymix.util.ghmm.wrapper import ARRAY_CALLOC, ARRAY_MALLOC


class dbackground:

    def __init__(self, n, m, orders, B):
        #  string ids of the background distributions
        self.name = ARRAY_CALLOC(n)
        for i in range(n):
            self.name[i] = None

        #  Number of distributions
        self.n = n
        #  Number of symbols in alphabet
        self.m = m
        #  Order of the respective distribution
        if orders:
            self.order = orders

        #  The probabilities
        if B:
            self.b = B


    def copy(self):
        new_order = ARRAY_MALLOC(self.n)
        new_b = ARRAY_CALLOC(self.n)

        for i in range(self.n):
            new_order[i] = self.order[i]
            b_i_len = pow(self.m, self.order[i] + 1)
            new_b[i] = ARRAY_CALLOC(b_i_len)
            for j in range(b_i_len):
                new_b[i][j] = self.b[i][j]

        tmp = dbackground(self.n, self.m, new_order, new_b)

        for i in range(self.n):
            if self.name[i]:
                tmp.name[i] = self.name[i]

        return tmp

    def setName(self, i, name):
        self.name[i] = name

    def getName(self, i):
        return self.name[i]

    def getOrder(self, i):
        return self.order[i]
