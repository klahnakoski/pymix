# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import unicode_literals
from __future__ import division

from datetime import datetime
from time import clock
from ..collections import MAX
from ..structs.wraps import wrap
from ..struct import Struct

ON = False
profiles = {}


class Profiler(object):
    """

    """

    def __new__(cls, *args):
        if ON:
            output = profiles.get(args[0], None)
            if output:
                return output
        output = object.__new__(cls, *args)
        return output

    def __init__(self, description):
        from ..queries.windows import Stats

        if ON and not hasattr(self, "description"):
            self.description = description
            self.samples = []
            self.stats = Stats()()
            profiles[description] = self

    def __enter__(self):
        if ON:
            self.start = clock()
        return self

    def __exit__(self, type, value, traceback):
        if ON:
            self.end = clock()
            duration = self.end - self.start

            from ..queries.windows import Stats

            self.stats.add(duration)
            if self.samples is not None:
                self.samples.append(duration)
                if len(self.samples) > 100:
                    self.samples = None


def write(profile_settings):
    from ..cnv import CNV
    from .files import File

    profs = list(profiles.values())
    for p in profs:
        p.stats = p.stats.end()

    stats = [{
        "description": p.description,
        "num_calls": p.stats.count,
        "total_time": p.stats.count * p.stats.mean,
        "total_time_per_call": p.stats.mean
    }
        for p in profs if p.stats.count > 0
    ]
    stats_file = File(profile_settings.filename, suffix=CNV.datetime2string(datetime.now(), "_%Y%m%d_%H%M%S"))
    if stats:
        stats_file.write(CNV.list2tab(stats))
    else:
        stats_file.write("<no profiles>")

    stats_file2 = File(profile_settings.filename, suffix=CNV.datetime2string(datetime.now(), "_series_%Y%m%d_%H%M%S"))
    if profs:
        r = range(MAX([len(p.samples) for p in profs]))
        profs.insert(0, Struct(description="index", samples=r))
        stats = [
            {p.description: wrap(p.samples)[i] for p in profs if p.samples}
            for i in r
        ]
        if stats:
            stats_file2.write(CNV.list2tab(stats))


