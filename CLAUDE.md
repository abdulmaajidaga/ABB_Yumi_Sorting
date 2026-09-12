# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

`yumi/` is the main workspace where new project code should be written. It is not itself a git
repository. `curobo/` is a vendored checkout of NVIDIA's cuRobo (a CUDA-accelerated robot motion
generation library — kinematics, IK, MPC, geometric planning, trajectory optimization, GPU-native
perception/ESDF, built on PyTorch/CUDA/Warp) and is its own git repo with its own toolchain — treat
it as a dependency to build against, not a place to casually make unrelated edits.

No project code exists in `yumi/` outside of `curobo/` yet — there is no root `pyproject.toml`,
`package.json`, or similar. When you start adding project files, set up the appropriate manifest at
`yumi/` root rather than nesting new work inside `curobo/`.

## Using curobo as a dependency

- The Python environment lives at `curobo/.venv` — activate it (`source curobo/.venv/bin/activate`)
  before running Python that imports curobo, or point a new project venv at it / install curobo
  into your own venv in editable mode from `curobo/`.
- Public API: import from `curobo.<module>` (e.g. `curobo.kinematics`, `curobo.inverse_kinematics`,
  `curobo.motion_planner`, `curobo.model_predictive_control`, `curobo.scene`, `curobo.types`) —
  these are the descriptive, user-facing modules. See `curobo/curobo/__init__.py` for the full
  public API list and a quick-start example.
- Do not import from `curobo._src` in project code — that's cuRobo's internal implementation layer
  (category-first-named modules like `optim/particle/optimizer_parallel_mppi.py`) and isn't a
  stable surface.
- Runtime behavior (CUDA graphs, torch.compile, kernel backend) is controlled by module-level flags
  in `curobo._src.runtime`, which must be set **before** constructing any cuRobo objects, e.g.:
  ```python
  import curobo._src.runtime as runtime
  runtime.cuda_graphs = True  # default; disable only for debugging/tests
  ```

## Working inside curobo/

If a task requires changing cuRobo itself (not just calling it), `curobo/AGENTS.md` is the entry
point for its own conventions (naming, testing, C++/CUDA style, docs style, DCO sign-off on
commits) — read it before editing files under `curobo/`. Key points if you end up there:

- Public API modules live at `curobo/curobo/*.py` (descriptive names); internal implementation
  lives under `curobo/curobo/_src/` (strict `category_specific.py` naming, general → specific).
- Tests mirror `_src/` 1:1 under `curobo/curobo/tests/_src/`. Run with `pytest` from `curobo/`
  (never use `float64` in tests, including gradcheck).
- Lint/format with `ruff check .` / `ruff format .`, type-check with `ty check`, both configured in
  `curobo/pyproject.toml`.
- Commits to curobo require a DCO sign-off: `git commit -s`.
