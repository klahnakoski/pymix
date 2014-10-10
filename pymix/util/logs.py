import re
from vendor.pyLibrary.env.logs import Log as moLog
from vendor.pyLibrary.struct import Struct


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
    def warning(template, *params):
        t, p = fix(template, params)
        moLog.warning(t, p)

    @staticmethod
    def error(template, *params):
        t, p = fix(template, params)
        moLog.error(t, p)


def fix(template, params):
    index = Struct(i=0)

    def asMoustache(match):
        output = "{{" + unicode(index.i) + "}}"
        index.i += 1
        return output

    t = re.sub(r"%\w", asMoustache, template, re.UNICODE)
    p = {unicode(i): v for i, v in enumerate(params)}
    return t, p



