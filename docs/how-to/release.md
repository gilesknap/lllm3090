# Cut a release

Releases are built and published by CI from a tag. Nothing is uploaded from a
laptop and there is no API token in the repository.

```bash
git tag 0.2.0
GIT_CONFIG_GLOBAL=/dev/null git -c credential.helper='!gh auth git-credential' \
  push https://github.com/gilesknap/lllm3090.git 0.2.0
```

The tag drives everything: `setuptools_scm` derives the version from it, so
there is no version string to edit and no way for the package version and the
tag to disagree.

On a tag push CI runs the tests, builds an sdist and wheel, checks them with
`twine check --strict`, installs the wheel and confirms `python -m lllm3090
--version` works — then publishes to PyPI.

## One-time PyPI setup

Publishing uses **trusted publishing** (OIDC), so PyPI verifies the workflow's
identity rather than a stored secret. Register the publisher at
<https://pypi.org/manage/account/publishing/> with:

| field | value |
|---|---|
| Owner | `gilesknap` |
| Repository | `lllm3090` |
| Workflow name | **`_pypi.yml`** |
| Environment | **`release`** |

:::{warning}
The workflow name must be `_pypi.yml`, **not** `ci.yml`. PyPI checks the OIDC
`job_workflow_ref` claim, and for a reusable workflow GitHub fills that in with
the file containing the job — not the entry point that called it. Registering
`ci.yml` makes every publish fail authentication at the tag, which is a
frustrating thing to discover during a release. The same applies if either the
file or the environment is ever renamed.
:::

Also create the `release` environment under the repository's Settings →
Environments. It needs no secrets; naming it is what lets the claim match.

## Installing a release

Once published, the package alone is a `pip install`:

```bash
pip install lllm3090
```

That gets the CLI and panel but not the engine or the service. `install.sh`
remains the path for a working system — it also fetches the pinned llama.cpp
build and installs the user unit.
