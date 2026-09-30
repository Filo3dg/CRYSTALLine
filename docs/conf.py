"""Sphinx configuration for the CRYSTALLine documentation.

The same theme CRYSTALClear's documentation uses (sphinx-book-theme), so the
two sites read as one family, with a small stylesheet of our own on top — see
_static/custom.css.

Pages are Markdown, through MyST. Build:

    sphinx-build -b html docs site
"""

import datetime
import pathlib
import re

project = "CRYSTALLine"
author = "@crystaldevs"   # the GitHub account, which is how the project signs
copyright = f"{datetime.datetime.now():%Y}, {author}"

# Read the version rather than repeat it: the release workflow already
# cross-checks pyproject.toml against src/crystalline/__init__.py, and a third
# copy here would be the one nobody remembers to bump.
_init = (pathlib.Path(__file__).parent.parent / "src" / "crystalline" / "__init__.py")
_match = re.search(r'^__version__\s*=\s*["\'](.+?)["\']', _init.read_text(), re.M)
release = version = _match.group(1) if _match else ""

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "attrs_inline",
    "attrs_block",
    "substitution",
]
myst_heading_anchors = 3

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"

# Developer notes that live in docs/ but are not pages of the site, and the
# build output if anyone points sphinx-build at the source tree.
exclude_patterns = ["CRYSTALClear_notes.md", "_build", "site", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_title = "CRYSTALLine"
html_logo = "logo.png"      # the light-mode file; see html_theme_options["logo"]
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_baseurl = "https://crystaldevs.github.io/CRYSTALLine/"

html_theme_options = {
    "announcement": (
        "CRYSTALLine is in beta. Errors are to be expected, and bug reports, "
        "suggestions, comments and requests for new features are all welcome: "
        "<a href='https://github.com/crystaldevs/CRYSTALLine/issues'>open an "
        "issue</a>."
    ),
    "repository_url": "https://github.com/crystaldevs/CRYSTALLine",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "home_page_in_toc": True,
    "navigation_with_keys": False,
    "show_navbar_depth": 1,
    "max_navbar_depth": 3,
    "logo": {
        "image_light": "logo.png",
        "image_dark": "_static/logo-dark.png",
    },
}

html_context = {
    # Light by default. The theme reads this from the context, not from
    # html_theme_options, where it is rejected as an unknown option.
    "default_mode": "light",
    "github_user": "crystaldevs",
    "github_repo": "CRYSTALLine",
    "github_version": "main",
    "doc_path": "docs",
}


# ── release notes, taken from the GitHub releases ─────────────────────────
# The notes are written once, on the release, and there is no reason to keep a
# second copy here that someone has to remember to update. The page is built
# from the API each time the site is. A build with no network still succeeds:
# the page then says where the notes are instead of carrying them, because a
# site that cannot be built offline is worse than one missing a page.
_RELEASES_API = "https://api.github.com/repos/crystaldevs/CRYSTALLine/releases"
_RELEASES_PAGE = "https://github.com/crystaldevs/CRYSTALLine/releases"
_RELEASES_FILE = "release-notes.md"


def _fetch_releases(timeout: float = 15.0):
    """The published releases, newest first, or ``None`` if they can't be read."""
    import json
    import os as _os
    import urllib.request

    request = urllib.request.Request(
        _RELEASES_API,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "CRYSTALLine-docs"},
    )
    # CI has a token and a much higher rate limit with it; locally there is none
    # and sixty requests an hour is plenty.
    token = _os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except Exception:  # noqa: BLE001 - offline, rate-limited, API changed: same answer
        return None


def _release_notes_page(releases) -> str:
    """The page, from what the API gave us — or the fallback when it gave nothing."""
    lines = ["# Release notes", ""]
    if not releases:
        lines += [
            "The notes for each release are published on GitHub:",
            "",
            f"[{_RELEASES_PAGE}]({_RELEASES_PAGE})",
            "",
            "(They are normally reproduced here, but the releases could not be "
            "read when this page was built.)",
            "",
        ]
        return "\n".join(lines)

    lines += [
        f"Every published release, newest first, as written on "
        f"[GitHub]({_RELEASES_PAGE}).",
        "",
    ]
    for release in releases:
        if release.get("draft"):
            continue
        tag = release.get("tag_name") or release.get("name") or "untagged"
        published = (release.get("published_at") or "")[:10]
        heading = f"## {tag}"
        if published:
            heading += f" — {published}"
        if release.get("prerelease"):
            heading += " (pre-release)"
        body = (release.get("body") or "").strip() or "_No notes were written for this release._"
        # A release note that starts its own headings must not outrank the page's.
        body = "\n".join(
            ("#" + line) if line.startswith("#") and not line.startswith("####") else line
            for line in body.splitlines()
        )
        lines += [heading, "", body, ""]
    return "\n".join(lines)


def _write_release_notes(app):
    from pathlib import Path as _Path

    page = _release_notes_page(_fetch_releases())
    (_Path(app.srcdir) / _RELEASES_FILE).write_text(page, encoding="utf-8")


def setup(app):
    app.connect("builder-inited", _write_release_notes)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
