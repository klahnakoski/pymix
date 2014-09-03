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

import copy
from .dataset import DataSet


class ConstrainedDataSet(DataSet):
    """
    Extension of the DataSet object that can hold pairwise or label constraints in the objects.
    This data set is required  for the semi-supervised learning EM
    """

    def __init__(self):
        DataSet.__init__(self)
        self.pairwisepositive = None
        self.pairwisenegative = None
        self.labels = []
        self.noLabels = 0

    def __copy__(self):
        cop = DataSet.__copy(self)
        cop.labels = copy.copy(self.labels)
        cop.pairwisepositive = copy.copy(self.pairwisepositive)
        cop.pairwisenegative = copy.copy(self.pairwisenegative)
        cop.noLabels = self.noLabels

    def setConstrainedLabels(self, labels):
        """
        Sets labels for semi-supervised learning.

        @param labels: list of lists of sample indices. The index in 'labels' denotes the component the samples are
        assigned to. For instance labels = [[0,2],[4,6]] would mean samples 0 and 2 are labelled with component 0.
        """
        assert sum([len(i) for i in labels]) <= self.N, 'Label constraints must be within the number of observations'
        self.labels = labels
        self.noLabels = len(labels)

    def removeFeatures(self, ids, silent=0):
        if self.labels:
            for i in ids:
                try:
                    index = self.headers.index(i)
                except ValueError:
                    pass
                for component in range(0, len(self.labels)):
                    component_lst = self.labels[component]
                    for element in range(0, len(component_lst)):
                        if component_lst[element] == index:
                            self.labels[component].pop(element)
                            break
        super(ConstrainedDataSet, self).removeFeatures(ids, silent)

    def removeSamples(self, ids, silent=0):
        if self.labels:
            indexlist = []
            for i in ids:
                newLabels = []
                try:
                    index = self.sampleIDs.index(i)
                    indexlist.append(index)

                except ValueError:
                    pass
                for component in self.labels:
                    newcomponent = []
                    if index in component:
                        component.remove(index)
                    for c in component:
                        if c < index:
                            newcomponent.append(c)
                        else:
                            newcomponent.append(c - 1)
                    newLabels.append(newcomponent)
                self.labels = newLabels
            indexlist.sort()
            indexlist.reverse()
            for index in indexlist:
                if self.pairwisenegative:
                    #print index, len(self.pairwisenegative)
                    self.pairwisenegative.pop(index)
                    for row in self.pairwisenegative:
                        row.pop(index)
                if self.pairwisepositive:
                    self.pairwisepositive.pop(index)
                    for row in self.pairwisepositive:
                        row.pop(index)
        super(ConstrainedDataSet, self).removeSamples(ids, silent)

    def setConstrainedLabelsFromColHeader(self):
        unique_classes = list(set(self.col_headers))

        for uc in unique_classes:
            class_list = []
            for index in range(0, len(self.col_headers)):
                if uc == self.col_headers[index]:
                    class_list.append(index)
            self.labels.append(class_list)

        self.noLabels = len(self.labels)

    def setConstrainedLabelsFromRowHeader(self):
        unique_classes = list(set(self.row_headers))

        for uc in unique_classes:
            class_list = []
            for index in range(0, len(self.row_headers)):
                if uc == self.row_headers[index]:
                    class_list.append(index)
            self.labels.append(class_list)

        self.noLabels = len(self.labels)

    def setPairwiseConstraints(self, positive, negative):
        """
        Set pairwise constraints.

        XXX add params
        """
        #if positive != None:
        #    assert len(positive) == self.p, 'Pairwise Constraints should cover the all observations'
        #if negative != None:
        #    assert len(negative) == self.p, 'Pairwise Constraints should cover the all observations'
        self.pairwisepositive = positive
        self.pairwisenegative = negative

    def getPairwiseConstraints(self, classes, prior_type):
        """
        @param data: classes of DataSet
        @param prior_type: 1 positive constr.
                                       2 negative constr.
        """
        prior_matrix = []
        for class1 in classes:
            prior_row = []
            for class2 in classes:
                if class1 == class2:
                    if prior_type == 1:
                        prior_row.append(1)
                    else:
                        prior_row.append(0)
                else:
                    if prior_type == 1:
                        prior_row.append(0)
                    else:
                        prior_row.append(1)
            prior_matrix.append(prior_row)
        return prior_matrix

