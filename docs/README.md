# BrainAgentSystem Documentation

## Getting Started

- [README](../README.md) - Quick start, installation, and basic usage
- [Architecture](architecture.md) - System design, 2047-node binary tree topology, multi-rate scheduling
- [API Reference](api.md) - Complete API documentation (Simulation, Experiment, I/O, Config, Testing)
- [Data Types](types.md) - BrainData hierarchy (Tensor, StateVector, Signal, SpikeTrain, Message)

## Design Documents

- [Design Template](DESIGN_TEMPLATE.md) - Template for designing new brain regions
- [Interface Spec](INTERFACE_SPEC.md) - Interface specifications for nodes and protocols

## Planning Documents

- [Improvement Plan](planning/IMPROVEMENT_PLAN.md) - Usability improvement tracking
- [Usability Plan](planning/USABILITY_PLAN.md) - Usability improvement plan details
- [Order Plan](planning/ORDER_PLAN.md) - Code organization and ordering plan
- [Original Plan](planning/PLAN.md) - Initial project plan
- [100 Phases Plan](planning/PLAN_100_PHASES.md) - Detailed 100-phase development plan

## Key Concepts

### Node Tree
The system models a brain as a 2047-node full binary tree (11 layers, L00-L10).
- **L00-L03** (15 nodes): Manually implemented with specific brain region logic
- **L04-L10** (2032 nodes): Auto-generated pass-through nodes

### Multi-Rate Scheduling
Three frequency bands drive the simulation:
- **Fast** (100Hz, dt=5ms): ASYNC_PUSH protocol, sensory/relay pathways
- **Medium** (20Hz, dt=20ms): SYNC_CALL protocol, motor/memory pathways
- **Slow** (1Hz, dt=100ms): PUB_SUB protocol, modulation systems

### Exception Hierarchy
All package errors inherit from `BrainAgentError`:
```
BrainAgentError
├── ConfigValidationError
├── NodeValidationError
├── TreeValidationError
├── SchedulerError → SchedulerConfigError
├── CheckpointError
├── SerializationError
├── InputError
└── ExportError
```
Each also inherits from the appropriate built-in (ValueError/RuntimeError) for backward compatibility.
