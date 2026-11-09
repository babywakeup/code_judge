"""C++ matcher with the existing Python configuration and result containers."""
from periodic_template_matcher.config import MatcherConfig


def __getattr__(name):
    if name in ('PeriodicTemplateMatcher','run_config'):
        from . import api
        return getattr(api,name)
    raise AttributeError(name)


__all__=['MatcherConfig','PeriodicTemplateMatcher','run_config']
