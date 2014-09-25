# encoding: utf-8
#
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
import os
import sys
from types import ModuleType

from ..jsons import json_encoder
from ..thread import threads
from ..struct import nvl, Struct, split_field, join_field
from ..structs.wraps import listwrap, wrap, wrap_dot
from ..strings import indent, expand_template
from ..thread.threads import Thread



DEBUG_LOGGING = False
ERROR = "ERROR"
WARNING = "WARNING"
UNEXPECTED = "UNEXPECTED"
NOTE = "NOTE"


class Log(object):
    """
    FOR STRUCTURED LOGGING AND EXCEPTION CHAINING
    """
    trace = False
    main_log = None
    logging_multi = None
    profiler = None   # simple pypy-friendly profiler
    cprofiler = None  # screws up with pypy, but better than nothing
    error_mode = False  # prevent error loops
    please_setup_constants = False  # we intend to manipulate module-level constants for debugging

    @classmethod
    def new_instance(cls, settings):
        settings = wrap(settings)

        if settings["class"]:
            if settings["class"].startswith("logging.handlers."):
                from .log_usingLogger import Log_usingLogger
                return Log_usingLogger(settings)
            else:
                try:
                    from .log_usingLogger import make_log_from_settings
                    return make_log_from_settings(settings)
                except Exception, e:
                    pass  # OH WELL :(

        if settings.log_type=="file" or settings.file:
            return Log_usingFile(file)
        if settings.log_type=="file" or settings.filename:
            return Log_usingFile(settings.filename)
        if settings.log_type=="stream" or settings.stream:
            from .log_usingStream import Log_usingStream
            return Log_usingStream(settings.stream)
        if settings.log_type=="elasticsearch" or settings.stream:
            from .log_usingElasticSearch import Log_usingElasticSearch
            return Log_usingElasticSearch(settings)

    @classmethod
    def add_log(cls, log):
        cls.logging_multi.add_log(log)

    @classmethod
    def debug(cls, template=None, params=None):
        """
        USE THIS FOR DEBUGGING (AND EVENTUAL REMOVAL)
        """
        Log.note(nvl(template, ""), params, stack_depth=1)

    @classmethod
    def println(cls, template, params=None):
        Log.note(template, params, stack_depth=1)

    @classmethod
    def note(cls, template, params=None, stack_depth=0):
        # USE replace() AS POOR MAN'S CHILD TEMPLATE

        log_params = Struct(
            template=template,
            params=nvl(params, {}).copy(),
            timestamp=datetime.utcnow(),
        )
        if cls.trace:
            log_template = "{{timestamp|datetime}} - {{thread.name}} - {{location.file}}:{{location.line}} ({{location.method}}) - " + template.replace("{{", "{{params.")
            f = sys._getframe(stack_depth + 1)
            log_params.location = {
                "line": f.f_lineno,
                "file": f.f_code.co_filename,
                "method": f.f_code.co_name
            }
            thread = Thread.current()
            log_params.thread = {"name": thread.name, "id": thread.id}
        else:
            log_template = "{{timestamp|datetime}} - " + template.replace("{{", "{{params.")

        cls.main_log.write(log_template, log_params)

    @classmethod
    def unexpected(cls, template, params=None, cause=None):
        if isinstance(params, BaseException):
            cause = params
            params = None

        if cause and not isinstance(cause, Except):
            cause = Except(UNEXPECTED, unicode(cause), trace=extract_tb(0))

        trace = extract_stack(1)
        e = Except(UNEXPECTED, template, params, cause, trace)
        Log.note(unicode(e), {
            "warning": {
                "template": template,
                "params": params,
                "cause": cause,
                "trace": trace
            }
        })


    @classmethod
    def warning(cls, template, params=None, cause=None):
        if isinstance(params, BaseException):
            cause = params
            params = None

        if cause and not isinstance(cause, Except):
            cause = Except(ERROR, unicode(cause), trace=extract_tb(0))

        trace = extract_stack(1)
        e = Except(WARNING, template, params, cause, trace)
        Log.note(unicode(e), {
            "warning": {  # REDUNDANT INFO
                "template": template,
                "params": params,
                "cause": cause,
                "trace": trace
            }
        })


    @classmethod
    def error(
            cls,
            template, # human readable template
            params=None, # parameters for template
            cause=None, # pausible cause
            offset=0        # stack trace offset (==1 if you do not want to report self)
    ):
        """
        raise an exception with a trace for the cause too
        """
        if params and isinstance(listwrap(params)[0], BaseException):
            cause = params
            params = None

        if cause == None:
            cause = []
        elif isinstance(cause, list):
            pass
        elif isinstance(cause, Except):
            cause = [cause]
        else:
            cause = [Except(ERROR, unicode(cause), trace=extract_tb(offset))]

        trace = extract_stack(1 + offset)
        e = Except(ERROR, template, params, cause, trace)
        raise e

    @classmethod
    def fatal(
        cls,
        template, # human readable template
        params=None, # parameters for template
        cause=None, # pausible cause
        offset=0    # stack trace offset (==1 if you do not want to report self)
    ):
        """
        SEND TO STDERR
        """
        if params and isinstance(listwrap(params)[0], BaseException):
            cause = params
            params = None

        if cause == None:
            cause = []
        elif isinstance(cause, list):
            pass
        elif isinstance(cause, Except):
            cause = [cause]
        else:
            cause = [Except(ERROR, unicode(cause), trace=extract_tb(offset))]

        trace = extract_stack(1 + offset)
        e = Except(ERROR, template, params, cause, trace)
        str_e = unicode(e)

        error_mode = cls.error_mode
        try:
            if not error_mode:
                cls.error_mode = True
                Log.note(str_e, {
                    "error": {
                        "template": template,
                        "params": params,
                        "cause": cause,
                        "trace": trace
                    }
                })
        except Exception, f:
            pass
        cls.error_mode = error_mode

        sys.stderr.write(str_e)


    # RUN ME FIRST TO SETUP THE THREADED LOGGING
    @classmethod
    def start(cls, settings=None):
        ## http://victorlin.me/2012/08/good-logging-practice-in-python/
        if not settings:
            return

        cls.settings = settings
        cls.trace = cls.trace | nvl(settings.trace, False)
        if cls.trace:
            from ..thread.threads import Thread

        if not settings.log:
            return

        cls.logging_multi = Log_usingMulti()
        cls.main_log = Log_usingThread(cls.logging_multi)

        for log in listwrap(settings.log):
            Log.add_log(Log.new_instance(log))

        if settings.cprofile:
            if isinstance(settings.cprofile, bool):
                settings.cprofile = {"enabled": True, "filename": "cprofile.tab"}

            import cProfile
            cls.cprofiler = cProfile.Profile()
            cls.cprofiler.enable()

        if settings.profile:
            from ..env import profiles

            if isinstance(settings.profile, bool):
                settings.profile = {"enabled": True, "filename": "profile.tab"}

            if settings.profile.enabled:
                profiles.ON = True

        if settings.constants:
            cls.please_setup_constants = True
        if cls.please_setup_constants:
            sys_modules = sys.modules
            # ONE MODULE IS MISSING, THE CALLING MODULE
            caller_globals = sys._getframe(1).f_globals
            caller_file = caller_globals["__file__"]
            if not caller_file.endswith(".py"):
                raise Exception("do not know how to handle non-python caller")
            caller_module = caller_file[:-3].replace("/", ".")

            for k, v in wrap_dot(settings.constants).leaves():
                module_name = join_field(split_field(k)[:-1])
                attribute_name = split_field(k)[-1].lower()
                if module_name in sys_modules and isinstance(sys_modules[module_name], ModuleType):
                    mod = sys_modules[module_name]
                    all_names = dir(mod)
                    for name in all_names:
                        if attribute_name == name.lower():
                            setattr(mod, name, v)
                    continue
                elif caller_module.endswith(module_name):
                    for name in caller_globals.keys():
                        if attribute_name == name.lower():
                            old_value = caller_globals[name]
                            try:
                                new_value = old_value.__class__(v)  # TRY TO MAKE INSTANCE OF SAME CLASS
                            except Exception, e:
                                new_value = v
                            caller_globals[name] = new_value
                            Log.note("Changed {{module}}[{{attribute}}] from {{old_value}} to {{new_value}}", {
                                "module": module_name,
                                "attribute": name,
                                "old_value": old_value,
                                "new_value": new_value
                            })
                            break
                else:
                    Log.note("Can not change {{module}}[{{attribute}}] to {{new_value}}", {
                        "module": module_name,
                        "attribute": k,
                        "new_value": v
                    })

    @classmethod
    def stop(cls):
        from ..env import profiles

        if cls.cprofiler and hasattr(cls, "settings"):
            write_profile(cls.settings.cprofile, cls.cprofiler)

        if profiles.ON and hasattr(cls, "settings"):
            profiles.write(cls.settings.profile)
        cls.main_log.stop()


    def write(self):
        raise NotImplementedError

def extract_stack(start=0):
    """
    SNAGGED FROM traceback.py
    Extract the raw traceback from the current stack frame.

    Each item in the returned list is a quadruple (filename,
    line number, function name, text), and the entries are in order
    from newest to oldest
    """
    try:
        raise ZeroDivisionError
    except ZeroDivisionError:
        trace = sys.exc_info()[2]
        f = trace.tb_frame.f_back

    for i in range(start):
        f = f.f_back

    stack = []
    n = 0
    while f is not None:
        stack.append({
            "depth": n,
            "line": f.f_lineno,
            "file": f.f_code.co_filename,
            "method": f.f_code.co_name
        })
        f = f.f_back
        n += 1
    return stack


def extract_tb(start):
    """
    SNAGGED FROM traceback.py

    Return list of up to limit pre-processed entries from traceback.

    This is useful for alternate formatting of stack traces.  If
    'limit' is omitted or None, all entries are extracted.  A
    pre-processed stack trace entry is a quadruple (filename, line
    number, function name, text) representing the information that is
    usually printed for a stack trace.
    """
    tb = sys.exc_info()[2]
    for i in range(start):
        tb = tb.tb_next

    trace = []
    n = 0
    while tb is not None:
        f = tb.tb_frame
        trace.append({
            "depth": n,
            "file": f.f_code.co_filename,
            "line": tb.tb_lineno,
            "method": f.f_code.co_name
        })
        tb = tb.tb_next
        n += 1
    trace.reverse()
    return trace


def format_trace(tbs, start=0):
    trace = []
    for d in tbs[start::]:
        d["file"] = d["file"].replace("/", "\\")
        item = expand_template('File "{{file}}", line {{line}}, in {{method}}\n', d)
        trace.append(item)
    return "".join(trace)


class Except(Exception):
    def __init__(self, type=ERROR, template=None, params=None, cause=None, trace=None):
        Exception.__init__(self)
        self.type = type
        self.template = template
        self.params = params
        self.cause = cause
        self.trace = trace

    @property
    def message(self):
        return unicode(self)

    def contains(self, value):
        if self.type == value:
            return True
        for c in self.cause:
            if c.contains(value):
                return True
        return False

    def __str__(self):
        output = self.type + ": " + self.template +"\n"
        if self.params:
            output = expand_template(output, self.params)

        if self.trace:
            output += indent(format_trace(self.trace))

        if self.cause:
            cause_strings = []
            for c in listwrap(self.cause):
                try:
                    cause_strings.append(unicode(c))
                except Exception, e:
                    pass

            output += "caused by\n\t" + "and caused by\n\t".join(cause_strings)

        return output

    def __unicode__(self):
        return unicode(str(self))

    def __json__(self):
        return json_encoder(Struct(
            type = self.type,
            template = self.template,
            params = self.params,
            cause = self.cause,
            trace = self.trace
        ))


class BaseLog(object):
    def write(self, template, params):
        pass

    def stop(self):
        pass


class Log_usingFile(BaseLog):
    def __init__(self, file):
        assert file

        from ..env.files import File

        self.file = File(file)
        if self.file.exists:
            self.file.backup()
            self.file.delete()

        self.file_lock = threads.Lock()

    def write(self, template, params):
        with self.file_lock:
            self.file.append(expand_template(template, params))




class Log_usingThread(BaseLog):
    def __init__(self, logger):
        # DELAYED LOAD FOR THREADS MODULE
        from ..thread.threads import Queue

        self.queue = Queue(max=10000, silent=True)
        self.logger = logger

        def worker(please_stop):
            while not please_stop:
                Thread.sleep(1)
                logs = self.queue.pop_all()
                for log in logs:
                    if log is Thread.STOP:
                        if DEBUG_LOGGING:
                            sys.stdout.write("Log_usingThread.worker() sees stop, filling rest of queue\n")
                        please_stop.go()
                    else:
                        self.logger.write(**log)

        self.thread = Thread("log thread", worker)
        self.thread.start()

    def write(self, template, params):
        try:
            self.queue.add({"template": template, "params": params})
            return self
        except Exception, e:
            sys.stdout.write("IF YOU SEE THIS, IT IS LIKELY YOU FORGOT TO RUN Log.start() FIRST\n")
            raise e  # OH NO!

    def stop(self):
        try:
            if DEBUG_LOGGING:
                sys.stdout.write("injecting stop into queue\n")
            self.queue.add(Thread.STOP)  # BE PATIENT, LET REST OF MESSAGE BE SENT
            self.thread.join()
            if DEBUG_LOGGING:
                sys.stdout.write("Log_usingThread telling logger to stop\n")
            self.logger.stop()
        except Exception, e:
            if DEBUG_LOGGING:
                raise e

        try:
            self.queue.close()
        except Exception, f:
            if DEBUG_LOGGING:
                raise f


class Log_usingMulti(BaseLog):
    def __init__(self):
        self.many = []

    def write(self, template, params):
        for m in self.many:
            try:
                m.write(template, params)
            except Exception, e:
                pass
        return self

    def add_log(self, logger):
        self.many.append(logger)
        return self

    def remove_log(self, logger):
        self.many.remove(logger)
        return self

    def clear_log(self):
        self.many = []

    def stop(self):
        for m in self.many:
            try:
                m.stop()
            except Exception, e:
                pass


def write_profile(profile_settings, cprofiler):
    from ..cnv import CNV
    from .files import File
    import pstats

    p = pstats.Stats(cprofiler)
    stats = [{
            "num_calls": d[1],
            "self_time": d[2],
            "total_time": d[3],
            "self_time_per_call": d[2] / d[1],
            "total_time_per_call": d[3] / d[1],
            "file": (f[0] if f[0] != "~" else "").replace("\\", "/"),
            "line": f[1],
            "method": f[2].lstrip("<").rstrip(">")
        }
        for f, d, in p.stats.iteritems()
    ]
    stats_file = File(profile_settings.filename, suffix=CNV.datetime2string(datetime.now(), "_%Y%m%d_%H%M%S"))
    stats_file.write(CNV.list2tab(stats))

if not Log.main_log:
    from log_usingStream import Log_usingStream
    Log.main_log = Log_usingStream("sys.stdout")




