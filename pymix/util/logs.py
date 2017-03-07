import re
from pyLibrary.debugs.logs import Log as moLog
from pyLibrary.dot import Dict


class Log(object):
    """
    CONVERT FROM C-LIKE TEMPLATES (WHICH USE %, WITH LIST OF PARAMETERS)
    TO MOUSTACHE (USING {{}}, AND A DICT OF PARAMETERS)

    THEN CALL NORMAL LOGGING
    """

    @staticmethod
    def note(template, *params):
        t, p = fix(template, params)
        moLog.note(t, p)

    @staticmethod
    def warning(*args, **kwargs):
        moLog.warning(*args, **kwargs)

    @staticmethod
    def error(*args, **kwargs):
        moLog.error(*args, **kwargs)


def fix(template, params):
    index = Dict(i=0)

    def asMoustache(match):
        output = "{{" + unicode(index.i) + "}}"
        index.i += 1
        return output

    t = re.sub(r"%\w", asMoustache, template, re.UNICODE)
    p = {unicode(i): v for i, v in enumerate(params)}
    return t, p



