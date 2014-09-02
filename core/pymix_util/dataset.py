import copy
import re
import numpy as np
import sys
from core.pymix_util.errors import InvalidDistributionInput
from core.parse import numerize
from core.parse import chomp, strTabList


class DataSet(object):
    """
        Class DataSet is the central data object.
    """

    def __init__(self):
        """
        Creates and returns an empty DataSet object
        """
        self.N = None   # number of samples
        self.p = None   # number of dimensions
        self.seq_p = None  # number of GHMM sequence features
        self.complex = None

        self.sampleIDs = []   # unique sample ids, row label
        self.headers = []     # label for each column
        self.dataMatrix = []
        self.col_headers = [] #ex:classes for each column
        self.row_headers = [] #ex:class for each row

        # attributes for internal data representation by sufficient statistics
        # these attributes are context specific (e.g. they depend on the MixtureModel)
        # and are initialised by the internalInit method.
        self.internalData = None

        self._internalData_views = None  # list of feature-wise views on internalData

        self.suff_dataRange = None
        self.suff_p = None
        self.suff_p_list = []


        # for each feature we can define a symbol (or value in case of continuous features)
        # to represent missing values in the data. This is used for instance in modelInitialization
        # to prevent the placeholders for missing data to influence the initial parameters.
        self.missingSymbols = {}


    def __len__(self):
        """
        Returns the number of samples in the DataSet.

        @return: Number of samples in the DataSet.
        """
        return self.N


    def __copy__(self):
        """
        Interface to copy.copy function.

        @return: deep copy of 'self'
        """

        cop = DataSet()

        cop.N = self.N
        cop.p = self.p
        cop.sampleIDs = copy.copy(self.sampleIDs)
        cop.row_headers = copy.copy(self.row_headers)
        cop.headers = copy.copy(self.headers)
        cop.col_headers = copy.copy(self.col_headers)
        cop.dataMatrix = copy.deepcopy(self.dataMatrix)
        cop.internalData = copy.deepcopy(self.internalData)

        cop._internalData_views = copy.deepcopy(self._internalData_views) # XXX

        cop.suff_dataRange = copy.copy(self.suff_dataRange)
        cop.suff_p = self.suff_p
        cop.suff_p_list = copy.deepcopy(self.suff_p_list)
        return cop

    def writeFile(self, filename):
        f = open(filename, 'w')
        if self.row_headers:
            f.write("ID\tNT\t")
        else:
            f.write("ID\t")
        f.write(strTabList(self.headers) + "\n")
        for i in range(self.N):
            if self.row_headers:
                f.write(str(self.sampleIDs[i]) + "\t" + str(self.row_headers[i]) + "\t" + strTabList(self.dataMatrix[i]) + "\n")
            else:
                f.write(str(self.sampleIDs[i]) + "\t" + strTabList(self.dataMatrix[i]) + "\n")
        f.close()


    def center(self, columns=None):
        dataAux = np.array(self.dataMatrix)
        if columns == None:
            columns = range(self.p)
        for c in columns:
            dataAux[:, c] = dataAux[:, c] - np.mean(dataAux[:, c])
        self.dataMatrix = dataAux.tolist()

    def ceil(self, value, columns=None):
        dataAux = np.array(self.dataMatrix)
        for i, line in enumerate(dataAux):
            for j, e in enumerate(line):
                dataAux[i, j] = min(e, value)
        self.dataMatrix = dataAux.tolist()


    def bottom(self, value, columns=None):
        dataAux = np.array(self.dataMatrix)
        for i, line in enumerate(dataAux):
            for j, e in enumerate(line):
                dataAux[i, j] = max(e, value)
        self.dataMatrix = dataAux.tolist()

    def transpose(self):
        if self.dataMatrix:
            dataAux = np.array(self.dataMatrix)
            dataAux = dataAux.transpose()
            self.dataMatrix = dataAux.tolist()

        aux = self.col_headers
        self.col_headers = self.row_headers
        self.row_headers = aux

        aux = self.headers
        self.headers = self.sampleIDs
        self.sampleIDs = aux

        self.N = len(self.dataMatrix)
        self.p = len(self.dataMatrix[0])

    def logTransform(self, columns=None, base=2):
        dataAux = self.dataMatrix
        #np.array(self.dataMatrix,dtype='Float64')
        if columns == None:
            columns = range(self.p)
        for d in dataAux:
            for c in columns:
                try:
                    if base == 2:
                        d[c] = str(np.log(float(d[c]) + 0.0000001))
                    else:
                        d[c] = str(np.log10(float(d[c]) + 0.000001))
                except ValueError:
                    pass
        self.dataMatrix = dataAux


    def foldChange(self, fold):
        dataAux = np.array(self.dataMatrix, dtype='Float64')
        samples = []
        for l in range(len(dataAux)):
            if sum(abs(dataAux[l, :]) >= fold):
                samples.append(self.sampleIDs[l])
        return samples


    def replace(self, column, values):
        dataAux = np.array(self.dataMatrix, dtype='Float64')
        dataAux[:, column] = values
        self.dataMatrix = dataAux.tolist()

    def normalize(self, columns=None, mad=0):
        dataAux = np.array(self.dataMatrix)
        if columns == None:
            columns = range(self.p)
        for c in columns:
            if mad:
                #print (np.median(dataAux[:,c]-np.median(dataAux[:,c])))
                dataAux[:, c] = dataAux[:, c] / (np.median(np.abs(dataAux[:, c] - np.median(dataAux[:, c]))))
            else:
                dataAux[:, c] = dataAux[:, c] / np.std(dataAux[:, c])
        self.dataMatrix = dataAux.tolist()


    def shuffle(self, columns=None):
        dataAux = np.array(self.dataMatrix)
        if columns == None:
            columns = range(self.p)
        for c in columns:
            np.random.shuffle(dataAux[:, c])
        self.dataMatrix = dataAux.tolist()

    def fromArray(self, array, IDs=None, headers=None, col_headers=None, row_headers=None):
        """
        Initializes the data set from a 'numpy' object.

        @param array: 'numpy' object containing the data
        @param IDs: sample IDs (optional)
        @param col_headers: feature headers (optional)
        """

        self.complex = 0  # DataSet is not complex
        self.seq_p = 0

        self.N = len(array)
        try:
            self.p = len(array[0])
        except TypeError:  # if len() raises an exception array[0] is not a list -> p = 1
            self.p = 1

        if not IDs:
            self.sampleIDs = range(self.N)
        else:
            self.sampleIDs = IDs

        if not headers:
            self.headers = range(self.p)
        else:
            self.headers = headers

        if col_headers:
            self.col_headers = col_headers
        if row_headers:
            self.row_headers = row_headers

        self.dataMatrix = array.tolist()


    def fromList(self, List, IDs=None, headers=None, col_headers=None, row_headers=None):
        """
        Initializes the data set from a Python list.

        @param List: Python list containing the data
        @param IDs: sample IDs (optional)
        @param col_header: feature headers (optional)
        """
        self.complex = 0  # DataSet is not complex
        self.seq_p = 0

        self.N = len(List)
        try:
            self.p = len(List[0])
        except TypeError:  # if len() raises an exception array[0] is not a list -> p = 1
            self.p = 1

        if IDs:
            self.sampleIDs = IDs
        else:
            self.sampleIDs = range(self.N)
        if headers:
            self.headers = headers
        else:
            self.headers = range(self.p)

        if col_headers:
            self.col_headers = col_headers
        if row_headers:
            self.row_headers = row_headers

        self.dataMatrix = List
        return self

    def fromFiles(self, fileNames, sep="\t", missing="*", fileID=None, IDheader=False, IDindex=None, silent=0):
        """
        Initializes the data set from a list of data flat files.

        @param fileNames: list of data flat files
        @param sep: separator string between values in flat files, tab is default
        @param missing: symbol for missing data '*' is default
        @param fileID: optional prefix for all features in the file
        @param IDheader: flag whether the sample ID column has a header in the first line of the flat files
        @param IDindex: index where the sample ids can be found, 0 by default
        """

        if IDindex == None:
            IDindex = [0] * len(fileNames)

        self.complex = 0  # DataSet is not complex
        self.seq_p = 0

        splitter = re.compile(sep)
        self.missingSymbols = [missing]

        # store data in dictionary with sampleIDs as keys
        data_dict = {}
        data_nrs = [0] * len(fileNames)
        for q, fname in enumerate(fileNames):
            f = open(fname, "r")

            # parse header line
            l1 = f.next().rstrip()

            l1 = l1.replace('"', '')  # remove " characters, if present

            # split at separation characters
            #list1= split(l1,sep)
            list1 = splitter.split(l1)

            # prepending file identifier to column labels
            if fileID:
                if silent == 0:
                    print "File ident: ", fileID
                for i in range(len(list1)):
                    list1[i] = str(fileID) + "-" + str(list1[i])

            if silent == 0:
                print fname, ":", len(list1), "features"

            #print list1

            if IDheader == True:  # remove header for sample ID column, if present
                tt = list1.pop(IDindex[q])
                #print tt
            #print list1

            data_nrs[q] = len(list1)
            self.headers = self.headers + list1

            for h, line in enumerate(f):
                line = chomp(line)
                line = line.replace('"', '')  # remove " characters, if present

                # cast data values into numerical types if possible
                sl = splitter.split(line)

                l = numerize(sl)
                sid = l.pop(IDindex[q])

                #print '->',sid

                if len(l) != data_nrs[q]:
                    print l
                    print list1
                    raise RuntimeError, "Different numbers of headers and data columns in files " + str(fname) + ", sample " + str(sid) + " ," + str(len(l)) + " != " + str(data_nrs[q])

                if not data_dict.has_key(sid):
                    data_dict[sid] = {}
                    data_dict[sid][fname] = l

                else:
                    data_dict[sid][fname] = l

        # assembling data set from data dictionary
        for k in data_dict:
            self.sampleIDs.append(k)
            citem = []
            for q, fname in enumerate(fileNames):
                if data_dict[k].has_key(fname):
                    citem += data_dict[k][fname]
                else:
                    incomplete = 1
                    print "Warning: No data for sample " + str(k) + " in file " + str(fname) + "."
                    citem += [missing] * data_nrs[q]

            self.dataMatrix.append(citem)

        self.N = len(self.dataMatrix)
        # checking data-label consistency
        for i in range(self.N):
            assert len(self.dataMatrix[i]) == len(self.headers), "Different numbers of headers and data columns in files " + str(fileNames) + ", sample " + str(self.sampleIDs[i]) + " ," + str(
                len(self.dataMatrix[i])) + " != " + str(len(self.headers))

        self.p = len(self.dataMatrix[0])

    def fromFile(self, fileName, sep='\t', col_headers=False, row_headers=False):

        f = open(fileName, "r")

        if row_headers:
            self.headers = f.next().rstrip().split(sep)[2:]
        else:
            self.headers = f.next().rstrip().split(sep)[1:]

        if col_headers:
            self.col_headers = f.next().rstrip().split(sep)[1:]

        for line in f:
            row = line.rstrip().split(sep)
            if len(row) == 0:
                break
            self.sampleIDs.append(row[0])
            if row_headers:
                self.row_headers.append(row[1])
                data_lin = [float(item) for item in row[2:]]
            else:
                data_lin = [float(item) for item in row[1:]]
            self.dataMatrix.append(data_lin)

        self.N = len(self.dataMatrix)

        # checking data-label consistency
        for i in range(self.N):
            assert len(self.dataMatrix[i]) == len(self.headers), "Different numbers of headers and data columns in files " + str(fileName) + ", sample " + str(self.sampleIDs[i]) + " ," + str(
                len(self.dataMatrix[i])) + " != " + str(len(self.headers))

        self.p = len(self.dataMatrix[0])

    def __str__(self):
        """
        String representation of the DataSet

        @return: string representation
        """
        strout = "Data set overview:\n"
        strout += "N = " + str(self.N) + "\n"
        strout += "p = " + str(self.p) + "\n\n"
        strout += "sampleIDs = " + str(self.sampleIDs) + "\n\n"
        strout += "row_headers = " + str(self.row_headers) + "\n\n"
        strout += "headers = " + str(self.headers) + "\n\n"
        strout += "column headers = " + str(self.col_headers) + "\n\n"
        #strout += "dataMatrix = "+str(self.dataMatrix) + "\n"

        return strout

    def printClustering(self, c, col_width=None):
        """
        Pretty print of a clustering .

        @param c: numpy array of integer cluster labels for each sample
        @param col_width: column width in spaces (optional)

        """
        if self.complex:
            raise NotImplementedError, "Needs to be done..."

        # get number of clusters in 'c'
        d = {}
        for lab in c:
            if lab == -1: # unassigned samples are handled seperately below
                continue
            d[lab] = ""
        G = len(d.keys())

        max_h = 0
        for h in self.headers:
            if len(str(h)) > max_h:
                max_h = len(str(h))
        max_sid = 0
        for s in self.sampleIDs:
            if len(str(s)) > max_sid:
                max_sid = len(str(s))
        if not col_width:
            space = max_h + 2
        else:
            space = col_width

        for i in d:
            t = np.where(c == i)
            index = t[0]
            print "\n----------------------------------- cluster ", i, "------------------------------------"
            print ' ' * (max_sid + 3),
            for k in range(len(self.headers)):
                hlen = len(str(self.headers[k]))
                print str(self.headers[k]) + " " * (space - hlen),
            print
            for j in range(len(index)):
                print '%-*s' % ( max_sid + 3, self.sampleIDs[index[j]]),
                for k in range(len(self.dataMatrix[index[j]])):
                    dlen = len(str(self.dataMatrix[index[j]][k]))
                    print str(self.dataMatrix[index[j]][k]) + " " * (space - dlen),
                print

        t = np.where(c == -1)
        index = t[0]
        if len(index) > 0:
            print "\n----------- Unassigned ----------------"
            space = max_h + 2
            print ' ' * (max_sid + 3),
            for k in range(len(self.headers)):
                hlen = len(str(self.headers[k]))
                print self.headers[k] + " " * (space - hlen),
            print
            for j in range(len(index)):
                print '%-*s' % ( max_sid + 3, self.sampleIDs[index[j]]),
                for k in range(len(self.dataMatrix[index[j]])):
                    dlen = len(str(self.dataMatrix[index[j]][k]))
                    print str(self.dataMatrix[index[j]][k]) + " " * (space - dlen),
                print

    def internalInit(self, m):
        """
        Initializes the internal representation of the data
        used by the EM algorithm .

        @param m: MixtureModel object
        """
        assert m.p == self.p, "Invalid dimensions in data and model." + str(m.p) + ' ' + str(self.p)

        templist = []
        for i in range(len(self.dataMatrix)):
            try:
                [t, dat] = m.components[0].formatData(self.dataMatrix[i])
            except InvalidDistributionInput, ex:

                # XXX broken in Python 2.6
                #ex.message += ' ( Sample '+str(self.sampleIDs[i])+', index = '+str(i)+' )'
                print ' ( Sample ' + str(self.sampleIDs[i]) + ', index = ' + str(i) + ' )'

                raise ex

            templist.append(dat)

        self.internalData = np.array(templist, dtype='Float64')

        if m.dist_nr > 1:
            self.suff_dataRange = copy.copy(m.components[0].suff_dataRange)
        else:
            self.suff_dataRange = [m.suff_p]

        self.suff_p = m.components[0].suff_p

        self._internalData_views = []
        for i in range(m.components[0].dist_nr):
            self.suff_p_list.append(m.components[0][i].suff_p)  # XXX suff_p_list should go away, only need in singleFeatureSubset
            if i == 0:
                prev_index = 0
            else:
                prev_index = self.suff_dataRange[i - 1]

            this_index = self.suff_dataRange[i]
            if self.p == 1:   # only a single feature
                self._internalData_views.append(self.internalData)
            else:
                self._internalData_views.append(self.internalData[:, prev_index:this_index])

    def getInternalFeature(self, i, m=None):
        """
        Returns the columns of self.internalData containing the data of the feature with index 'i'

        @param i: feature index
        @param m: mixture model
        @return: numpy containing the data of feature 'i'
        """
        if self.suff_dataRange is None:
            self.internalInit(m)

        #assert self.suff_dataRange is not None,'DataSet needs to be initialized with .internalInit()'
        if i < 0 or i >= len(self.suff_dataRange):
            raise IndexError, "Invalid index " + str(i)

        return self._internalData_views[i]

    def removeFeatures(self, ids, silent=0):
        """
        Remove a list of features from the data set.

        @param ids: list of feature identifiers
        @param silent: verbosity control
        """
        # removing columns from data matrix

        for i in ids:
            try:
                ind = self.headers.index(i)
            except ValueError:
                sys.stderr.write("\nERROR:  Feature ID " + str(i) + " not found.\n")
                raise

            for k in range(self.N):
                self.dataMatrix[k].pop(ind)

            r = self.headers.pop(ind)
            if self.col_headers:
                self.col_headers.pop(ind)
            self.p -= 1
            if not silent:
                print "Feature " + str(r) + " has been removed."

    def extractFeatures(self, ids, silent=0):
        """
        Removes all features not in the list from the data set.

        @param ids: list of feature identifiers to keep
        @param silent: verbosity control
        """
        # removing columns from data matrix


        allids = copy.deepcopy(self.headers)
        allids.reverse()
        for i in allids:
            if i in ids:
                continue
            try:
                ind = self.headers.index(i)
            except ValueError:
                sys.stderr.write("\nERROR:  Feature ID " + str(i) + " not found.\n")
                raise

            for k in range(self.N):
                self.dataMatrix[k].pop(ind)

            r = self.headers.pop(ind)
            if self.col_headers:
                self.col_headers.pop(ind)
            self.p -= 1
            if not silent:
                print "Feature " + str(r) + " has been removed."

    def filterMissing(self, percentage):
        """
        Remove all samples with more than percentage of missing values

        @param percentage: minimal allowed missing value percentage
        """
        dataAux = np.array(self.dataMatrix)
        missing = np.zeros((1, len(dataAux)))
        for symbol in self.missingSymbols:
            missing = np.sum(dataAux == symbol, axis=1)
            #print sum(missing), np.mean(missing)
        remove = []
        #print missing
        self.missingStatistics()
        for i, j in enumerate(missing):
            #print  j/float(len(dataAux[0])), percentage
            if j / float(len(dataAux[0])) > percentage:
                remove.append(self.sampleIDs[i])
        self.removeSamples(remove)
        dataAux = np.array(self.dataMatrix)
        for symbol in self.missingSymbols:
            missing = np.sum(dataAux == symbol, axis=1)
        print "Missing After"
        self.missingStatistics()
        print "Samples", len(remove), len(self.sampleIDs)

    def missingStatistics(self):
        """Statitics of missing values
        """
        dataAux = np.array(self.dataMatrix)
        missing = np.zeros((1, len(dataAux)))
        for symbol in self.missingSymbols:
            missing = np.sum(dataAux == symbol, axis=1)
            #print "Missing Entries", sum(missing)
        #print "Genes with Missing Entries", sum(missing>0)
        #print missing
        return sum(missing), sum(missing > 0)

    def replaceMissing(self, median=0):
        """
        Replace missing symbols with mean value of feature
        """
        dataRes = np.zeros((len(self.dataMatrix), len(self.dataMatrix[0])))
        for i, line in enumerate(self.dataMatrix):
            for j, element in enumerate(line):
                try:
                    dataRes[i, j] = float(element)
                except ValueError:
                    dataRes[i, j] = 'nan'
        nanvalues = np.isnan(dataRes)
        dataRes = np.ma.masked_array(dataRes, np.isnan(dataRes))
        if median:
            mean = np.median(dataRes, axis=1)
        else:
            mean = np.mean(dataRes, axis=1)
        for i in range(len(dataRes)):
            dataRes[i, nanvalues[i, :]] = mean[i]
        self.dataMatrix = dataRes.tolist()

    def filterFold(self, fold, novariables):
        """
        Remove samples that do not show a value higher (lower) than fold
        at least in novariables

        @param fold fold change expected
        @param novariable number of variables where fold change occurs

        @return filtered samples

        """
        dataAux = np.array(self.dataMatrix, dtype='Float64')

        #means = np.mean(dataAux,axis=0)
        #for i,m in enumerate(means):
        #    dataAux[:,i]=dataAux[:,i]
        #print means
        changes = np.sum(abs(dataAux) >= fold, axis=1)
        print changes
        remove = []
        ids = []
        for i, j in enumerate(changes):
            #print i, j
            if j < novariables:
                remove.append(self.sampleIDs[i])
                ids.append(i)
        print 'removed samples', len(remove)
        self.removeSamples(remove)
        return remove


    def removeSamples(self, ids, silent=0):
        """
        Remove a list of samples from the data set.

        @param ids: list of sample identifiers
        @param silent: verbosity control
        """

        if self.internalData != None:
            print "Warning: internalInit has to be rerun after removeSamples."
            self.internalData = None
            self.suff_dataRange = None
            self.suff_p = None
            self.suff_p_list = []

        rmcount = 0   # count number of samples actually removed
        for si in ids:
            try:
                sind = self.sampleIDs.index(si)
            except ValueError:  # sample is already not in DataSet
                continue
            self.dataMatrix.pop(sind)
            self.sampleIDs.pop(sind)
            if self.row_headers:
                self.row_headers.pop(sind)
            rmcount += 1

        if not silent:
            print str(rmcount) + ' samples removed'

        self.N = self.N - rmcount

    def extractSamples(self, ids, silent=0):
        """
        Removes all samples not in input list from the data set.

        @param ids: list of sample identifiers
        @param silent: verbosity control
        """
        if self.internalData:
            print "Warning: internalInit has to be rerun after removeSamples."
            self.internalData = None
            self.suff_dataRange = None
            self.suff_p = None
            self.suff_p_list = None

        count = 0

        cp_sid = copy.copy(self.sampleIDs)
        for si in cp_sid:
            if si in ids:
                continue
            try:
                sind = self.sampleIDs.index(si)
            except ValueError:  # sample is already not in DataSet
                continue
            self.dataMatrix.pop(sind)
            self.sampleIDs.pop(sind)
            if self.row_headers:
                self.row_headers.pop(sind)
            count += 1

        if not silent:
            print str(self.N - count) + ' samples retained'

        self.N = self.N - count

    def filterSamples(self, fid, min_value, max_value):
        """
        Removes all samples with values < 'min_value' or > 'max_value' in feature 'fid'.

        @param fid: feature ID in self.headers
        @param min_value: minimal required value
        @param max_value: maximal required value

        """
        if self.internalData:
            print "Warning: internalInit has to be rerun after removeSamples."
            self.internalData = None
            self.suff_dataRange = None
            self.suff_p = None
            self.suff_p_list = None

        ind = self.headers.index(fid)

        print "Removing samples with " + fid + " < " + str(min_value) + " or > " + str(max_value) + " ...",
        i = 0  # current index in dataMatrix
        c = 0  # number of samples already considered
        r = 0  # number of removed samples
        while c < self.N:
            if self.dataMatrix[i][ind] < min_value or self.dataMatrix[i][ind] > max_value:
                # remove sample
                self.dataMatrix.pop(i)
                self.sampleIDs.pop(i)
                if self.row_headers:
                    self.row_headers.pop(i)
                c += 1
                r += 1
            else:
                i += 1
                c += 1

        print str(r) + " samples removed"
        self.N = self.N - r


    def maskDataSet(self, valueToMask, maskValue, silent=False):
        """
        Allows the masking of a value with another in the entire data matrix.

        @param valueToMask: value to be masked
        @param maskValue: value which is to be substituted
        @param silent: verbosity control (False is default)
        """
        count = 0
        for i in range(self.N):
            for j in range(self.p):
                if self.dataMatrix[i][j] == valueToMask:
                    self.dataMatrix[i][j] = maskValue
                    count += 1

        if not silent:
            print str(count), "values '" + str(valueToMask) + "' masked with '" + str(maskValue) + "' in all features."


    def maskFeatures(self, headerList, valueToMask, maskValue):
        """
        Equivalent to maskDataSet but constrained to a subset of features

        @param headerList: list of features IDs
        @param valueToMask: value to be masked
        @param maskValue: value which is to be substituted
        """
        count = 0
        for h in headerList:
            try:
                ind = self.headers.index(h)
            except ValueError:
                sys.stderr.write("\nERROR:  Feature ID " + str(h) + " not found.\n")
                raise

            for j in range(self.N):
                if str(self.dataMatrix[j][ind]) == str(valueToMask):
                    self.dataMatrix[j][ind] = maskValue
                    count += 1

        print str(count), "values '" + str(valueToMask) + "' masked with '" + str(maskValue) + "' in " + str(len(headerList)) + " features."

    def discretizeFeature(self, fid, bins, missing='*'):
        """
        Discretizes the values for a given feature.

        @param fid: feature ID
        @param bins: number of levels for discretization
        @param missing: optional missing symbol

        """
        j = self.headers.index(fid)
        # get minimal / maximal value
        vmax = float('-inf')
        vmin = float('inf')
        for i in range(self.N):
            if self.dataMatrix[i][j] == missing:
                continue

            if self.dataMatrix[i][j] > vmax:
                vmax = self.dataMatrix[i][j]
            if self.dataMatrix[i][j] < vmin:
                vmin = self.dataMatrix[i][j]
            # discretize
        step = (vmax - vmin) / (float(bins) - 1)
        for i in range(self.N):
            if self.dataMatrix[i][j] == missing:
                continue
            self.dataMatrix[i][j] = int(self.dataMatrix[i][j] / step)
        #            print 'v=',self.dataMatrix[i][j],' ->',self.dataMatrix[i][j]/step,' ->',int(self.dataMatrix[i][j]/step)


    def getExternalFeature(self, fid):
        """
        Returns the external data representation of a given feature

        @param fid: feature ID in self.headers

        @return: list of data samples for feature fid
        """
        index = self.headers.index(fid)
        res = []
        for i in range(self.N):
            res.append(self.dataMatrix[i][index])

        return res

    def extractSubset(self, ids):
        """
        Remove all samples in 'ids' from 'self' and return a new DataSet initialised with these samples

        @param ids: list of sample indices

        @return: DataSet object containing the samples in ids
        """
        res = DataSet()

        res.N = len(ids)
        res.p = self.p
        res.suff_dataRange = copy.copy(self.suff_dataRange)
        res.suff_p = self.suff_p
        res.suff_p_list = self.suff_p_list
        res.sampleIDs = ids
        res.headers = copy.copy(self.headers)
        res.col_headers = copy.copy(self.col_headers)
        res.row_headers = []

        if self.internalData is not None:
            res.internalData = np.zeros((res.N, res.suff_p), self.internalData.type())

        else:
            res.internalData = None

        #remove subset entries from self.internalData
        new_intData = None
        if self.internalData is not None:
            new_intData = np.zeros(( (self.N - res.N), res.suff_p), self.internalData.type())
            #new_sampleIDs = []
            ni = 0
            for i in range(self.N):
                if self.sampleIDs[i] not in ids:
                    new_intData[ni] = self.internalData[i]
                    ni += 1

        for i, d in enumerate(ids):
            ind = self.sampleIDs.index(d)

            dat = self.dataMatrix.pop(ind)
            res.dataMatrix.append(dat)

            # fill internalData matrix
            if self.internalData is not None:
                res.internalData[i] = copy.deepcopy(self.internalData[ind])

            self.sampleIDs.pop(ind)
            if self.row_headers:
                res.row_headers.append(self.row_headers[ind])

        self.internalData = new_intData
        self.N -= res.N

        return res

    def singleFeatureSubset(self, index):
        """
        Returns a DataSet for the feature with internal index 'index' in 'self'.
        For internal use.

        @param index: feature index

        @return: DataSet object
        """
        res = DataSet()

        res.N = self.N    # number of samples
        res.p = 1    # number of dimensions
        res.seq_p = self.seq_p  # number of GHMM sequence features
        res.suff_p = self.suff_p_list[index]
        res.suff_p_list = [self.suff_p_list[index]]
        res.internalData = self.getInternalFeature(index)
        res.col_headers = []
        res.row_headers = []
        res._internalData_views = [self._internalData_views[index]] # XXX

        if self.headers:
            res.headers.append(self.headers[index])
        if self.col_headers:
            res.col_headers.append(self.col_headers[index])
        if self.sampleIDs:
            res.sampleIDs = self.sampleIDs
        if self.row_headers:
            res.row_headers = self.row_headers

        res.missingSymbols = {}
        if self.missingSymbols.has_key(index):
            res.missingSymbols[0] = self.missingSymbols[index]

        res.suff_dataRange = [self.suff_dataRange[index] - self.suff_dataRange[index - 1]]
        return res

    def setMissingSymbols(self, findices, missing):
        """
        Assigns missing value placeholders to features.

        @param findices: list of internal feature indices
        @param missing: list of missing symbols/values
        """
        assert len(findices) == len(missing)
        for i, h in enumerate(findices):
            self.missingSymbols[h] = missing[i]

    def getMissingIndices(self, ind):
        """
        Get indices of missing values in one feature

        @param ind: feature index

        @return: list of indices of missing values
        """
        assert self.suff_dataRange is not None

        if not self.missingSymbols.has_key(ind):
            return []
        else:
            m_ind = []
            dat_ind = self.getInternalFeature(ind)
            for i, v in enumerate(dat_ind):
                # check sample 'i' for missing symbol
                if np.all(v == self.missingSymbols[ind]):
                    m_ind.append(i)
            return m_ind


    def writeClusteringFasta(self, fn_pref, m):
        """
        Writes a clustering based on model 'm' into files in FASTA format.
        Note that this implies sequence data.

        @param fn_pref: Filename prefix. The full name of each output file consists
        of the prefix, the cluster number and the extension .fa
        @param m: MixtureModel object
        """
        c = m.classify(self, silent=1)

        # get number of clusters in 'c'
        d = {}
        for a in c:
            d[a] = ""
        G = len(d)

        # write data to file
        for k in d:
            t = np.where(c == k)
            index = t[0]
            f = open(fn_pref + '_' + str(k) + '.fa', 'w')
            for i in index:
                f.write(">" + str(self.sampleIDs[i]) + "\n")
                s = ''.join(self.dataMatrix[i]) # convert to sequence
                f.write(s + "\n")
            f.close()

