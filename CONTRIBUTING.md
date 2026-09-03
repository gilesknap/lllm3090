# Contributing

Contributions are welcome, including the small ones: a model that works, a
speed measured on a card nobody here owns, a paragraph of documentation that
was wrong.

## Set up

```bash
git clone https://github.com/gilesknap/lllm3090.git
cd lllm3090
uv venv
uv pip install -e ".[dev]"
```

`uv`, not `pip`: the lockfile is `uv.lock`, and the installed CLI on this
project's own machine comes from `uv tool install`. Plain `pip install -e
".[dev]"` inside a virtualenv works too if that is what you have.

You do **not** need a GPU to work on this. The whole test suite runs without
one — `hardware.detect()` reports `present=False` and the catalogue is computed
against the reference card so it can still be read — and CI has no GPU either.
What needs a card is measuring speed, and that is exactly the contribution
nobody else can make for you.

## The three things CI runs

```bash
ruff check src tests
pytest -q
sphinx-build -W -b html docs docs/_build/html
```

Run all three before opening a pull request. The third one catches more than
people expect: `-W` turns Sphinx warnings into errors, so an unresolvable
cross-reference in a docstring, or a colon in the first line of an attribute
docstring, fails the build.

`mypy src` is clean and worth keeping that way, though CI does not currently
gate on it.

## Style

Ruff settings live in `pyproject.toml`: 88 columns, `B`, `C4`, `E`, `F`, `I`,
`UP`, `W`. Beyond that, the convention in this codebase is that **comments say
why, not what**. Most of the comments here exist because something was
surprising once — an engine that was being killed by systemd on every panel
restart, a window that was silently 39k tokens too small — and the comment is
what stops it being surprising twice. If you fix something that took a while to
understand, leave the reason behind.

Docstrings are in the same voice. They are rendered into the API reference by
Sphinx, so they are prose, not type restatements.

## Tests

New logic comes with tests. The suite runs in about twenty seconds and needs
nothing but a Python interpreter: no GPU, no engine, no network. Keep it that
way — a download test serves its bytes from a `ThreadingHTTPServer` on
loopback, and an engine test fakes `Popen` rather than launching anything.

Coverage is uneven by design rather than by neglect: the catalogue arithmetic
is what decides whether a model fits, so it is tested hardest.

`tests/conftest.py` points every path in `config` — models, engines, state — at
an empty temporary directory before each test, so the suite sees the same
machine on your laptop as it does in CI. It is not a tidiness measure: before
it existed, choosing the CUDA engine once from the panel left a stored choice
that made 33 unrelated tests fail, and only on a machine that had been used.

Two tests opt back out with `@pytest.mark.real_machine`, because checking the
declared MTP flags against the real weights and the real engines against
`--list-devices` is the whole point of them. Both skip themselves where those
files are absent, which is why CI stays green without them. If you add a third,
it skips too.

## Adding a model to the catalogue

1. Add the entry to `src/lllm3090/data/models.yaml`. See
   [the catalogue reference](https://gilesknap.github.io/lllm3090/reference/catalogue.html)
   for what each field means.
2. `lllm3090 models` — check the computed context and fit look sane.
3. `pytest -q` — the catalogue tests check the arithmetic for every entry.
4. If you have run it, `lllm3090 bench <name>` prints a profile block to
   include. Speeds in this project are measured, never estimated: an entry with
   no measurement says so rather than guessing, and a figure measured on
   another card is labelled as such.

## Pull requests

One issue per commit where it divides that way, with `Fixes #n` in the commit
message, so a reviewer can read a slice and a release can trace a change back
to the reason for it. Commit messages here carry the reasoning — what was
happening before, why the obvious fix was not the one taken — rather than a
restatement of the diff.

Add a line to `CHANGELOG.md` under `## [Unreleased]` if the change is one a
user would notice.
