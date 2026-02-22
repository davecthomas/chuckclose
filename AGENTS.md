# **LLM Code prompt guidance**

# **Coding preferences**

## **Language & runtime**

* Python \>= **3.11.13**.

* US English everywhere (identifiers, comments, errors, logs).

## Project Structure

Use a flat project structure (files in root) for this waitlist shell project.
Refrain from forcing a `src/` layout unless the project grows significantly.
Tests should be in a `tests/` directory if they exist. 

## .env

Keep env\_template.txt with any env vars and include helpful comments about default values and uses of these.   
Ensure reading of any env vars always has helpful defaults except in the case of true secrets.

## File Naming convention

Try to name new files based on the primary class or feature contained therein. Avoid generic names.   
Example: if the class is AIMetricsProviderBase, the file should be named ai\_metrics\_provider\_base.py

## Security

Never use “random” to generate a random number. Use secrets. 

## Variables

\- **Note**: Existing code in this project uses standard Python `snake_case` without Hungarian notation. Preserve existing style in legacy files.
\- For **new** code, the following preferences apply:
\- Use descriptive names prefixed by type for non-primitive objects (e.g., \`dict_settings\`, \`list_dict_teams\`, \`random_index\`).  
\- Primitive objects do not need prefixes, except for strings, which should have “str_” prefix.  
\- Instances of classes should have a helpful prefix that indicates the class name it is an instance of.   
\- Do not abbreviate words and never use one-letter variable names (e.g., prefer \`speed\`, \`dict_voices\`).

## Type hints

* **Everything is typed**: parameters and return values.
* **Local variables**: Explicit typing is preferred but type inference is acceptable where obvious.

* **PEP 604 unions** (str | None) — never Optional\[str\].

* Prefer concrete, precise types; use Any only at **external boundaries** (e.g., boto3 payloads).

## Typed containers & generics

* Use **built‑in generics**: list\[str\], dict\[str, Any\], set\[str\], tuple\[str, int\].

   Rationale: modern, concise, and the standard in 3.11+.

* **Never** use legacy typing.List, typing.Dict, typing.Set, typing.Tuple.

   Rationale: deprecated style; hurts readability.

* For **fixed‑length tuples** use tuple\[T1, T2\]; for homogeneous variable length use tuple\[T, ...\].

   Rationale: communicates shape vs sequence.

* When a function only **reads** a collection, prefer **ABCs** \- Accept `Sequence[T]`/`Mapping[K, V]` for read-only.   
* Accept `MutableSequence[T]`/`MutableMapping[K, V]` only when **mutation** is required.

* Import ABCs from **collections.abc** (e.g., from collections.abc import Iterable, Sequence, Mapping, Callable).

   Rationale: they’re the canonical runtime generics in modern Python.

* Callable types are written as Callable\[\[ArgType1, ArgType2\], ReturnType\].

   Rationale: explicit signatures aid IDEs and code review.

* For structured dicts with a fixed schema, use **Pydantic V2** models to validate and coerce data 

## Class and object property or method access

Ensure you know class fields and their types. Do not use getattr or hasattr. It demonstrates you are lacking an understanding of the code context. 

## Structure & readability

* Move **slowly**, step‑by‑step; confirm the plan **before** edits.

* No nested functions; flat, testable helpers.

* Assign to a named variable before returning (helps devs with debugging); avoid dense inline returns.

* Keep module‑level side effects minimal (no work at import time beyond constants).

## **Naming conventions**

* Do **not** rename existing variables unless required to complete the task, such as if the purpose of the variable is materially changed

* Names must help clarify the type or role of the var in context (e.g., list\_subnet\_ids, map\_headers, set\_arns, dict\_user\_properties).  
  Rationale: improves readability and reduces dev errors.

## **3rd-party APIs**

* Always wrap 3rd-party API calls with a general factory pattern where we  
1. Create a ABC class for the type of API provider   
2. Subclass to create a wrapper class for the specific 3rd party API   
3. Use a factory.create\_client class to instantiate the specific subclass based on a registry (e.g. .env)  
* For 3rd party API calls, always attempt to discern retryable from non-retryable errors. Create generic methods within the API wrapper class to test for retryable errors (e.g. keep a list of specific exceptions and refer back to this list) and retry-with-backoff functionality

## **Logging, errors, and retries**

* Wrap any critical calls with try to catch all relevant exceptions. Ensure exception objects are caught and logged   
* Use structured, concise log lines; **no timestamps** in messages  
* Prefer %s parameterized logging (lazy formatting) over f‑strings in log calls.  
* Error messages must be actionable; wrap low‑level exceptions with context.  
* Never just raise an error without an appropriate-level log message.   
* forbid secrets/PII in log messages  
* Don’t just give up on errors. Decide if they can be retried or not before deciding on logging and error handling steps

## Comment and documentation rules

* Always fully comment module, class, method, and any calls to 3rd party APIs which often require explanations for specific parameter usage  
* Never remove existing comments unless they are no longer relevant.  
* Reword or expand comments for clarity.  
* US English only  
* Add comments any place where an important decision occurs.  
* When using third-party APIs, provide a helpful API documentation URL in the comments.  
* Use only ASCII characters in comments and documentation; avoid special Unicode characters that may render poorly in VSCode.

## Module import practices

- Enforce import ordering (standard library → third-party → local) with `isort` in pre-commit.  
- **Do not** purge modules from `sys.modules`; avoid patterns such as:

```py
  import sys
  sys.modules.pop("yaml", None)
```

## **HTTP API Resiliency**

* Implement **bounded exponential backoff** on known transient API endpoint errors (e.g. catch specific AWS API exceptions)

* Validate inputs **before** calling APIs (fail fast with 4xx‑style errors when appropriate).

## **Output & change management**

* **Never provide diffs** unless specifically requested. Why: it’s nearly impossible to copy/paste this into an IDE. 

* Always deliver **full code blocks** (drop‑in, production‑ready) at the function or class level (depending on the task) unless a surgical edit is requested. 

* Keep comments focused on **why**, not restating code.

## **Library and config management**

The project uses `requirements.txt` and `venv` for dependency management.
Ensure `requirements.txt` is kept up to date. 

## **Environment Restrictions**

* **Do not run `poetry install`, `pip install`, or other network-dependent commands.**
* Assume the user manages the environment and dependencies.
* Rely on `poetry run` for executing scripts if the environment is known to be set up, but avoid installation commands. 

## **Production Code ONLY**

Unless specifically requested for pseudocode \- NEVER provide exemplars with incomplete code or assumptions based on hallucinated system features. Only produce full production-ready code.

## Versioning

* Bump the version number whenever external input changes or output behavior changes—even subtle voice-setting tweaks.  
* After a bump, update `README.md`, `pyproject.toml`, and `__version__.py`.

## Commits

* Begin each commit message with the new version tag for context, e.g., `v1.2.0 updates foo for bar.`

## Branch naming

* If a JIRA ticket or GitHub issue is referenced, prefix the branch name with that identifier.
