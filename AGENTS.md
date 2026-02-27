# Repository guidelines

## Python version

We are using python 3.11.13 so use all modern python language constructs supported as of that version.

## 1 · Formatting & style

- Keep code formatted with `black` (default settings). **CRITICAL:** Before completing any code changes or responding to the user that a fix is complete, you MUST run `poetry run black .` and `poetry run ruff check .`. If Ruff reports issues, you MUST fix them (for example, run `poetry run ruff check . --fix` when appropriate) and re-run `poetry run ruff check .` until all checks pass.
- Use US English for all documentation and comments.
- Include type hints for every variable, parameter, and return value.
- When using Pydantic, always use Pydantic v2.

## 2 · Directory layout

- Place all source files and examples in `src/upside_lib_ai_api_unified/`.
- Store all tests in the `tests/` directory and mock network calls whenever possible.

## 3 · Variables

- Always use PEP 604 union-operator syntax (`Type | None`) for optional types instead of `Optional[Type]`.
- Never use typing types from prior to PEP 585 (e.g. avoid Dict, List, Tuple, etc)

#### Variable naming

- Use descriptive names prefixed by type for non-primitive objects (e.g., `dict_settings`, `list_dict_teams`, `random_index`).
- Do not abbreviate words and never use one-letter variable names. Exceptions to this include any widely used abbreviations, such as "config" for configuration, "env" for environment, "idx" for index, "id" for identifier, "str" for string, "int" for integer, "float" for float, "bool" for boolean, etc.

#### No hardcoded strings or numbers in variable assignments

- Create descriptive constant names for any variable assignments or logical tests for clarity.

## NEVER use getattr or hasattr if you know the class methods and properties can be accessed

getattr and hasattr is hard to read and unnecessary if you know the class members.

## Comment and documentation rules

- Never remove existing comments unless they are no longer relevant.
- Feel free to reword or expand comments for clarity.
- **CRITICAL:** Every single function MUST include a comprehensive docstring describing:
  1. What the function does (its purpose).
  2. Each input parameter's use and type constraints.
  3. The exact output structure and any potential return variations.
- **CRITICAL:** ALL loops must have a comment at the beginning describing what is being accomplished in the loop.
- **CRITICAL:** All function returns must be commented inline, explaining why the return is happening (e.g. 'Normal exit', or 'Early return due to error').
- Add comments to class headers, new logical blocks, and any place where an important decision occurs., such as a function return prior to a regular function exit.
- When using third-party APIs, provide a helpful API documentation URL in the comments.
- Use only ASCII characters in comments and documentation; avoid special or “fancy” Unicode characters that may render poorly in VSCode.

## 5 · Code modification constraints

- Do **not** reformat code that is unrelated to the prompt request (including whitespace or line-break changes).
- Do **not** rename variables unless required by a change in data type.

## 6 · Module import practices

- Enforce import ordering (standard library → third-party → local) with `isort` in pre-commit.
- **Do not** purge modules from `sys.modules`; avoid patterns such as:

```python
  import sys
  sys.modules.pop("yaml", None)
```

## 7 · Versioning

- Bump the version number whenever external input changes or output behavior changes—even subtle voice-setting tweaks. If the API is changing, bump minor version.
- After a bump, update `README.md`, `pyproject.toml`, and `__version__.py`.

## 8 · Commits

- Begin each commit message with the new version tag for context, e.g., `v1.2.0 updates foo for bar.`

## 9 · Branch naming

- If a JIRA ticket or GitHub issue is referenced, prefix the branch name with that identifier.

## Logging

Use a shared logger rather than printing info, warning, errors.
