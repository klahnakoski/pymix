################################################################################
#
#       This file is part of the Modified Python Mixture Package, original
#       source code is from https://svn.code.sf.net/p/pymix/code.  Also see
#       http://www.pymix.org/pymix/index.php?n=PyMix.Download
#
#       Changes made by: Kyle Lahnakoski (kyle@lahnakoski.com)
#
################################################################################
#
#       This file is part of the Python Mixture Package
#
#       file:    mixture.py
#       author: Benjamin Georgi
#
#       Copyright (C) 2004-2009 Benjamin Georgi
#       Copyright (C) 2004-2009 Max-Planck-Institut fuer Molekulare Genetik,
#                               Berlin
#
#       Contact: georgi@molgen.mpg.de
#
#       This library is free software; you can redistribute it and/or
#       modify it under the terms of the GNU Library General Public
#       License as published by the Free Software Foundation; either
#       version 2 of the License, or (at your option) any later version.
#
#       This library is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#       Library General Public License for more details.
#
#       You should have received a copy of the GNU Library General Public
#       License along with this library; if not, write to the Free
#       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
################################################################################

import tokenize
import cStringIO
from pymix.util.emission_domain import Alphabet


#----------------------------------- File IO --------------------------------------------------
# XXX  While functional the flat file based file IO is somewhat crude. XXX
# XXX  The whole thing ought to be redone in XML at some point.        XXX

def numerize(data):
    """
    Cast all elements in a list to numeric values.

    @param data: list of data

    @return: list of processed data
    """

    for i in range(len(data)):
        try:
            data[i] = int(data[i])

        except ValueError:
            try:
                data[i] = float(data[i])
            except ValueError:
                pass
    return data


def writeMixture(model, fileName, silent=False):
    """
    Stores model parameters in file 'fileName'.

    @param model: MixtureModel object
    @param fileName: file name the model is to be written to
    """
    from pymix.models.bayes import BayesMixtureModel
    from pymix.models.mixture import MixtureModel
    from pymix.models.labeled_bayes import labeledBayesMixtureModel

    f = open(fileName, 'w')
    if isinstance(model, labeledBayesMixtureModel):
        head = 'labelBayesMix'
    elif isinstance(model, BayesMixtureModel):
        head = "BayesMix"
    elif isinstance(model, MixtureModel):
        head = "Mix"
    else:
        raise TypeError

    if not model.struct:
        l = str(';' + head) + ";" + str(model.G) + ";" + str(model.pi.tolist()) + ";" + str(model.compFix) + "\n"
    else:
        l = str(';' + head) + ";" + str(model.G) + ";" + str(model.pi.tolist()) + ";" + str(model.compFix) + ";" + str(model.leaders) + ";" + str(model.groups) + "\n"

    f.write(l)
    for i in range(model.G):
        l = model.components[i].flatStr(0)
        f.write(l)

    if head == "BayesMix" or head == 'labelBayesMix':
        l = model.prior.flatStr(0)
        f.write(l)

    if not silent:
        print "Model written to file " + str(fileName) + "."
    f.close()


def readMixture(fileName):
    """
    Reads model from file 'fileName'.

    @param fileName: file to be read

    @return: MixtureModel object
    """
    f = open(fileName, 'r')
    s = chomp(f.readline())
    struct = 0
    if len(s.split(';')) == 5:  # MixtureModel object
        [offset, head, G, pi, compFix] = s.split(';')
        leaders = None
        groups = None
    elif len(s.split(';')) == 7:  # BayesMixtureModel object
        struct = 1
        [offset, head, G, pi, compFix, leaders, groups] = s.split(';')
    else:
        raise IOError, 'Flat file format not recognized.'
    if leaders and groups:
        mixModel = parseMix(f, head, int(G), simple_eval(pi), simple_eval(compFix), simple_eval(leaders), simple_eval(groups))
    else:
        mixModel = parseMix(f, head, int(G), simple_eval(pi), simple_eval(compFix))
    f.close()
    return mixModel


def parseMix(fileHandle, mtype, G, pi, compFix, leaders=None, groups=None):
    """
    Parses a flat file for a mixture model. Internal function, is invoked from
    readMixture.

    """
    components = []
    while len(components) < G:
        components.append(parseFile(fileHandle))

    if mtype == 'Mix':
        from pymix.models.mixture import MixtureModel
        m = MixtureModel(G, pi, components, compFix=compFix)

    elif mtype == 'labelBayesMix':
        from pymix.models.labeled_bayes import labeledBayesMixtureModel
        prior = parseFile(fileHandle)
        if sum(compFix) > 0: # XXX pass compFix if it is not trivial
            m = labeledBayesMixtureModel(G, pi, components, prior, compFix=compFix)
        else:
            m = labeledBayesMixtureModel(G, pi, components, prior)

    elif mtype == 'BayesMix':
        from pymix.models.bayes import BayesMixtureModel
        prior = parseFile(fileHandle)
        if sum(compFix) > 0: # XXX pass compFix if it is not trivial
            m = BayesMixtureModel(G, pi, components, prior, compFix=compFix)
        else:
            m = BayesMixtureModel(G, pi, components, prior)

    else:
        raise TypeError
    if leaders and groups:
        m.initStructure()
        m.leaders = leaders
        m.groups = groups
        for i in range(m.dist_nr):
            for lead in m.leaders[i]:
                for g in m.groups[i][lead]:
                    if not m.components[lead][i] == m.components[g][i]:
                        raise IOError, 'Incompatible CSI structure and parameter values in parseMix.'
                    m.components[g][i] = m.components[lead][i]
    return m


def parseProd(fileHandle, true_p):
    """
    Internal function. Parses product distribution.
    """
    from pymix.distributions.product import ProductDistribution

    distList = []
    p = 0
    while p < true_p:
        d = parseFile(fileHandle)
        distList.append(d)
        p += d.p
    return ProductDistribution(distList)


def parseMixPrior(fileHandle, nr_dist, structPrior, nrCompPrior):
    from pymix.priors.mixture_model import MixtureModelPrior
    c = []
    piPrior = parseFile(fileHandle)
    for i in range(nr_dist):
        p = parseFile(fileHandle)
        c.append(p)
    return MixtureModelPrior(structPrior, nrCompPrior, piPrior, c)


def parseDirichletMixPrior(fileHandle, G, M, pi):
    from pymix.priors.dirichlet_mixture import DirichletMixturePrior

    dC = [parseFile(fileHandle) for i in range(G)]
    return DirichletMixturePrior(G, M, pi, dC)


def parseFile(fileHandle):
    """
    Internal function. Parses flat files.
    """
    s = chomp(fileHandle.readline())
    l = s.split(';')

    if l[1] == "Mix":
        [offset, head, G, pi, compFix] = l
        return parseMix(fileHandle, head, int(G), simple_eval(pi), simple_eval(compFix))
    elif l[1] == "Norm":
        from pymix.distributions.normal import NormalDistribution
        [offset, head, mu, sigma] = l
        return NormalDistribution(float(mu), float(sigma))
    elif l[1] == "Exp":
        from pymix.distributions.exponential import ExponentialDistribution
        [offset, head, lambd] = l
        return ExponentialDistribution(float(lambd))
    elif l[1] == "Mult":
        from pymix.distributions.multinomial import MultinomialDistribution
        [offset, head, N, M, phi, alphabet, parFix] = l
        alph = Alphabet(simple_eval(alphabet))
        return MultinomialDistribution(int(N), int(M), simple_eval(phi), alph, simple_eval(parFix))
    elif l[1] == "Discrete":
        from pymix.distributions.discrete import DiscreteDistribution
        [offset, head, M, phi, alphabet, parFix] = l
        alph = Alphabet(simple_eval(alphabet))
        return DiscreteDistribution(int(M), simple_eval(phi), alph, simple_eval(parFix))
    elif l[1] == "MultiNormal":
        from pymix.distributions.multinormal import MultiNormalDistribution
        [offset, head, p, mu, sigma] = l
        # XXX the tokenize package used in simple_eval cannot deal with negative values in
        # mu or sigma. A hack solution to that would be to change simple_eval to a direct
        # call to eval in the line below. This carries all the usual implications for security.
        return MultiNormalDistribution(int(p), simple_eval(mu), simple_eval(sigma))
    elif l[1] == "Dirichlet":
        from pymix.distributions.dirichlet import DirichletDistribution
        [offset, head, M, alpha] = l
        return DirichletDistribution(int(M), simple_eval(alpha))
    elif l[1] == "DirichletPr":
        from pymix.priors.dirichlet import DirichletPrior
        [offset, head, M, alpha] = l
        return DirichletPrior(int(M), simple_eval(alpha))
    elif l[1] == "NormalGamma":
        from examples.crp import NormalGammaPrior
        [offset, head, mu, kappa, dof, scale] = l
        return NormalGammaPrior(float(mu), float(kappa), float(dof), float(scale))
    # elif l[1] == "PriorForDirichlet":
    #     [offset, head, M, eta] = l
    #     return PriorForDirichletDistribution(int(M), simple_eval(eta))
    elif l[1] == "Prod":
        [offset, head, p] = l
        return parseProd(fileHandle, int(p))
    elif l[1] == "MixPrior":
    #;MixPrior;4;0.7;0.7
        [offset, head, nr_dist, structPrior, nrCompPrior] = l
        return parseMixPrior(fileHandle, int(nr_dist), float(structPrior), float(nrCompPrior))
    elif l[1] == "DirichMixPrior":
        #;DirichMixPrior;3;5;[ 0.3  0.3  0.4]
        [offset, head, G, M, pi] = l
        return parseDirichletMixPrior(fileHandle, int(G), int(M), simple_eval(pi))
    else:
        raise TypeError, "Unknown keyword: " + str(l[1])


def chomp(string):
    """
    Removes a newline character from the end of the string if present

    @param string: input string

    @return: the argument without tailing newline.

    """
    if string[-1] == "\n" or string[-1] == "\r":
        return string[0:-1]
    #    elif string[len(string)-4:] == "<cr>":
    #        return string[0:-4]
    else:
        return string

# The following functions come courtesy of the people at comp.lang.python
def sequence(next, token, end):
    out = []
    token = next()
    while token[1] != end:
        out.append(atom(next, token))
        token = next()
        if token[1] == "," or token[1] == ":":
            token = next()
    return out


def atom(next, token):
    if token[1] == "(":
        return tuple(sequence(next, token, ")"))
    elif token[1] == "[":
        return sequence(next, token, "]")
    elif token[1] == "{":
        seq = sequence(next, token, "}")
        res = {}
        for i in range(0, len(seq), 2):
            res[seq[i]] = seq[i + 1]
        return res
    elif token[0] in (tokenize.STRING, tokenize.NUMBER):
        return eval(token[1]) # safe use of eval!
    raise SyntaxError("malformed expression (%s)" % token[1])


def simple_eval(source):
    src = cStringIO.StringIO(source).readline
    src = tokenize.generate_tokens(src)
    src = (token for token in src if token[0] is not tokenize.NL)
    res = atom(src.next, src.next())
    if src.next()[0] is not tokenize.ENDMARKER:
        raise SyntaxError("bogus data after expression")
    return res


def strTabList(List):
    stres = str(List[0])
    for elem in List[1:]:
        stres += "\t" + str(elem)
    return stres
