
import textwrap

def copydoc(fromfunc, sep="\n"):
    """
    Decorator: Copy the docstring of `fromfunc`
    """
    def _decorator(func):
        sourcedoc = fromfunc.__doc__
        if func.__doc__ is None:
            func.__doc__ = sourcedoc
        else:
            func.__doc__ = sep.join([
                textwrap.dedent(sourcedoc),
                textwrap.dedent(func.__doc__),
            ])
        return func
    return _decorator

