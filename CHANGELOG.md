# Changelog

## [0.5.0] - 2026-06-20

### Added
- Simulation.run() now supports on_step callback for progress feedback
- Simulation.iter_steps() generator API for fine-grained control
- tqdm progress bar support (progress=True parameter)
- Simulation context manager support (__enter__/__exit__)
- ResultExporter context manager support
- Simulation.get_node_strict() with helpful error messages
- brain_agent.testing module with MockNode, RecordingNode, build_small_tree
- BrainConfig.to_yaml() and save() methods
- CLI subcommands: run, info, validate
- console_scripts entry point: brain-agent command
- --version flag
- examples/06_custom_node.py - custom node implementation tutorial
- examples/07_experiment_sweep.py - batch experiment and parameter sweep

### Fixed
- Environment variable override bug: BRAIN_SIMULATION_DT_FAST now correctly maps to simulation.dt_fast
- Error messages now list valid options (e.g., "Supported formats: json, csv, npz")
- All error messages translated to English for consistency
- Simulation.export() no longer auto-closes for memory exports
- Error messages include file paths where relevant

### Changed
- CLI restructured with subcommands (run/info/validate)
- Version number now read from package metadata (single source of truth)
