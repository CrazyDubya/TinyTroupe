# TinyTroupe dev log (supplements git commits)

## 2026-02-15 — Repo & branch audit

### Remote branches (origin = https://github.com/CrazyDubya/TinyTroupe.git)

| Branch | Latest commit | Description |
|--------|---------------|-------------|
| `main` | d98598b | feat: memory size limits (Phase 1, Task 1.1) — **current default** |
| `fix-loglevel-initialization` | 301f6e0 | Fix: Apply initial log level from config at startup |
| `sync-upstream-nov-2025` | 9644c76 | Merge upstream microsoft/TinyTroupe main |
| `claude/implement-phase-one-01C4EgchBzF9vnG2z1qNdNcf` | 19d7f6e | feat: semantic similarity caching (Phase 1, Task 3.3) |
| `claude/prep-major-expansion-018HXYCcspXCWe3e3M7Wvg81` | 116ed79 | docs: Fix review feedback on expansion documentation |

### Local clone

- **Path:** `/Users/pup/TinyTroupe`
- **Checked-out:** `main`
- **Status:** Clean, up to date with `origin/main`

### Repo copies / iCloud

- **iCloud:** No TinyTroupe folder in `~/Library/Mobile Documents/com~apple~CloudDocs/`
- **car-hub:** TinyTroupe not present in `/Users/pup/car-hub/`
- **Other clones:** None found (single working copy at `/Users/pup/TinyTroupe`)

### Branch ancestry (for canonical discussion)

- `main` has advanced beyond `sync-upstream-nov-2025`; merge-base: 91a6632
- `sync-upstream-nov-2025` merged upstream `microsoft/TinyTroupe` main
- Other branches are feature/docs branches, not separate forks

---

## 2026-02-15 — Canonical consolidation

- Brought in unique content from iCloud **tinyexperiment** and **prison** clones
- Created `experiment_archives/tinyexperiment/` with docs, simulation scripts, business/infrastructure/quality design, result JSONs
- Added `TheatreCompany.ipynb`, `aethelburg_scenario/`, `theatre_tools.py` to `examples/`
- Documented prison as upstream baseline (no unique files; merged via sync-upstream)
- Created `EXPERIMENTS_INDEX.md` and `experiment_archives/prison/README.md`
- **main** is the canonical branch
- Second commit: added CommunityRallyTool.py, UnemployedYouth.agent.json to examples/aethelburg_scenario

### Next steps completed (2026-02-15)

1. **Pushed main** to origin
2. **Removed redundant iCloud clones:** github_other/TinyTroupe, tinyexperiment/tinytroupe, prison/TinyTroupe
3. **Experiments reference:** README now links to EXPERIMENTS_INDEX.md; use it as the reference for what's where
