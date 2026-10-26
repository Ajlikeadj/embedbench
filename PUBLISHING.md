# Publishing embedbench to PyPI

The package is already PyPI-ready: hatchling build backend, single-sourced version,
`py.typed`, sdist + wheel, `twine check` clean, and project URLs pointing at
<https://github.com/Ajlikeadj/embedbench>. This guide covers the two supported ways to
upload it, plus the rules that bite on a first release.

> At the time of writing, `embedbench` is unclaimed on both PyPI and TestPyPI, which is
> what lets you create the project. Re-check before relying on that.

---

## 0. Pre-flight checklist

1. **Bump the version** in `src/embedbench/__init__.py`. It is the single source of truth
   (`[tool.hatch.version]` reads it) and also lands in `embedbench.__version__`.
2. **Add a CHANGELOG entry** for the version you are about to publish.
3. **Build and validate locally:**

   ```powershell
   Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
   .\.venv\Scripts\python.exe -m pip install --upgrade build twine
   .\.venv\Scripts\python.exe -m build
   .\.venv\Scripts\python.exe -m twine check dist\*
   ```

   Expected: `embedbench-X.Y.Z-py3-none-any.whl` and `embedbench-X.Y.Z.tar.gz`, both
   `PASSED`.
4. **Confirm the metadata** you are about to freeze forever:

   ```powershell
   .\.venv\Scripts\python.exe -c "import zipfile;print(zipfile.ZipFile('dist/embedbench-0.2.0-py3-none-any.whl').read('embedbench-0.2.0.dist-info/METADATA').decode())"
   ```

   Check `Name`, `Version`, `Requires-Python`, `Project-URL`, `Provides-Extra`.
5. **TestPyPI dry run (recommended for the first upload).** Upload there first, install
   from there, then upload to PyPI. Both indexes are independent; the TestPyPI upload does
   not reserve your name on PyPI.

> **Irreversible:** a version/filename can never be replaced, even after deleting or
> yanking a release. Any fix requires a version bump.

---

## Option A — Trusted Publishing from GitHub Actions (recommended)

No PyPI API token is stored anywhere. GitHub proves the workflow's identity to PyPI with
OIDC, using the workflow in [`.github/workflows/publish.yml`](.github/workflows/publish.yml).

### A1. Create the pending publisher on PyPI

Open <https://pypi.org/manage/account/publishing/> → **Add a new pending publisher** and
fill in exactly:

| Field | Value |
| --- | --- |
| PyPI project name | `embedbench` |
| Owner | `Ajlikeadj` |
| Repository name | `embedbench` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

For TestPyPI, repeat at <https://test.pypi.org/manage/account/publishing/> with the same
values except **Environment name** = `testpypi`.

The pending publisher is what allows the very first upload to create the project; after
that it becomes a normal trusted publisher and can be managed under the project settings.

### A2. Create the GitHub environments

Repository → **Settings → Environments**:

- `pypi`
- `testpypi` (only if you plan to dry-run)

Optionally add required reviewers to `pypi` so a release needs a human approval.

### A3. Trigger a publish

Two triggers are configured:

- **Publish a GitHub Release** → `release: published` runs the workflow. Create the release
  for a tag whose commit contains the version you want (e.g. `v0.2.1`).
- **Manual run** → Actions → **Publish** → *Run workflow* → choose `pypi` or `testpypi`.
  This is the simplest way to publish the version currently on `main` without cutting a
  new tag.

### A4. Verify

```bash
pip install embedbench==<version>
python -c "import embedbench; print(embedbench.__version__)"
```

Project page: <https://pypi.org/project/embedbench/>

---

## Option B — Upload manually from your machine

### B1. Account and token

1. Create an account at <https://pypi.org/account/register/> and enable 2FA (mandatory).
2. Create an API token at <https://pypi.org/manage/account/token/>.
   - **First upload:** the project does not exist yet, so use an **account-scoped** token.
   - **After the first upload:** create a token scoped to the `embedbench` project and
     revoke the account-scoped one.

Never commit the token, never paste it into a chat or issue, and do not put it in
`pyproject.toml`. Prefer the environment-variable form below over `~/.pypirc`.

### B2. Upload

PowerShell:

```powershell
# from the repository root
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m twine check dist\*

$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-AgEIcHlwaS5vcmc..."   # paste your token here, never commit it
.\.venv\Scripts\python.exe -m twine upload dist\*

Remove-Item Env:TWINE_PASSWORD                       # clear it afterwards
```

bash/zsh:

```bash
rm -rf dist && python -m build && python -m twine check dist/*
TWINE_USERNAME=__token__ TWINE_PASSWORD='pypi-...' python -m twine upload dist/*
```

The username must be the literal `__token__`; the password is the token (including its
`pypi-` prefix).

### B3. TestPyPI first (optional but recommended)

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-AgEIcHlwaS5vcmc..."     # a test.pypi.org token, not the PyPI one
$env:TWINE_REPOSITORY_URL = "https://test.pypi.org/legacy/"
.\.venv\Scripts\python.exe -m twine upload dist\*
Remove-Item Env:TWINE_REPOSITORY_URL

# install the dry-run build
.\.venv\Scripts\python.exe -m pip install --index-url https://test.pypi.org/simple/ `
  --extra-index-url https://pypi.org/simple/ embedbench==0.2.0
```

TestPyPI resolves dependencies from TestPyPI, which hosts only packages uploaded there;
`--extra-index-url` lets pip fetch NumPy/Pydantic/etc. from real PyPI.

---

## Versioning rules

- One version = one upload. Bump `__version__` for every change you publish.
- Tag releases as `vX.Y.Z`. Tags and PyPI versions should agree; publish the release from
  the tagged commit.
- Yanking a release hides it from resolvers but keeps the version consumed, and the
  filename cannot be reused.

Suggested release loop:

```bash
# 1. edit src/embedbench/__init__.py and CHANGELOG.md
# 2. verify
python -m build && python -m twine check dist/*
# 3. commit + tag
git add -A && git commit -m "release: vX.Y.Z"
git tag -a vX.Y.Z -m "embedbench X.Y.Z"
git push origin main --tags
# 4. publish: GitHub Release (auto workflow) or twine upload
```

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `403 Invalid or non-existent authentication information` | Wrong username (must be `__token__`) or wrong/expired token. |
| `400 File already exists` | That version was already uploaded. Bump the version; you cannot overwrite. |
| `400 ... Invalid classifier` / `License-Expression` | Fix `pyproject.toml`, rebuild, retry with a new version if already uploaded. |
| `403 The user 'embedbench' isn't allowed to upload to project` | Token is scoped to another project; create one for `embedbench`. |
| Trusted publishing: `invalid-publisher` / OIDC error | Pending publisher mismatch: owner, repo, workflow filename (`publish.yml`), or environment name (`pypi`). It must match exactly and the workflow must exist on the default branch. |
| `No matching distribution found` after TestPyPI upload | You installed without `--extra-index-url https://pypi.org/simple/`. |
| Package installs but `embedbench` command missing | The console script ships in the wheel only; reinstall from PyPI, not from a partial checkout. |

---

## Current state of this repository

The GitHub release `v0.2.0` was tagged at commit `5ac70ab`. The project-URL metadata fix
in `pyproject.toml` landed afterwards on `main`, and PyPI metadata comes from the built
artifact, not the git tag. So:

- **To publish 0.2.0 as-is:** build from `main` (URLs correct), then run the Publish
  workflow manually or `twine upload`. The version string is still `0.2.0`.
- **If you want tag and artifact to match exactly:** cut `v0.2.1` (bump the version) and
  publish that instead.
