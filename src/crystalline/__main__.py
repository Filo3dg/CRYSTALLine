"""Entry point: ``python -m crystalline`` (or the ``crystalline`` command).

``--check`` exists because the installed command is a ``gui_scripts`` entry: on
Windows that runs under ``pythonw`` with no console at all, so a startup failure
prints to nothing and the command looks like it did nothing. A user with a
problem needs one thing they can run that talks back.
"""

from __future__ import annotations

import sys

_USAGE = """\
usage: crystalline [file] [--check] [--version]

  file       a CRYSTAL .out/.gui/.f34 or a .cif to open on startup
  --check    check this machine has what CRYSTALLine needs, and say what is
             missing if not
  --version  print the version and exit
"""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--help" in argv or "-h" in argv:
        print(_USAGE)
        return 0
    if "--version" in argv:
        from crystalline import __version__

        print(__version__)
        return 0
    if "--check" in argv or "--doctor" in argv:
        from crystalline.diagnose import report

        return report()

    _leave_a_trace_on_a_native_crash()

    from crystalline.app import run

    return run()


def _leave_a_trace_on_a_native_crash() -> None:
    """Make a crash that is not a Python error leave something behind.

    A fault in VTK, Qt or a C extension kills the process without raising, so
    the log the launcher keeps simply ends mid-sentence, with nothing to say
    where it stopped — which is exactly what a crash during an animation export
    looked like. ``faulthandler`` prints the Python stack of every thread to
    stderr when the signal arrives, and the launcher is already sending stderr
    to ~/Library/Logs/CRYSTALLine.log.

    It covers SIGSEGV, SIGABRT, SIGBUS, SIGFPE and SIGILL. A process the system
    kills for using too much memory gets SIGKILL, which nothing can catch: the
    silence itself is then the evidence, which is worth knowing when reading a
    log that stops without a word.
    """
    import faulthandler

    try:
        faulthandler.enable()
    except Exception:  # noqa: BLE001 - a stderr it cannot write to is not fatal
        pass


if __name__ == "__main__":
    sys.exit(main())
