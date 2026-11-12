# Native registry format fixtures

These are 13 synthetic tensor programs serialized by the original TenSet TVM
fork at commit `35774ed373f9693cc6efcbe83f5703141e8ab0c8`. They contain no
TenSet benchmark records, timing measurements, or experimental results.

- `native_registry.pkl`: actual native `SearchTask` objects, pickle protocol 4.
- `native_reference.json`: independently extracted tensor/axis/stage metadata
  and SHA-256 values for the exact native SaveJSON graphs.
- `generate_native.py`: reproducible definitions and reference generator.
- `validate_native.py`: development-only comparison of portable metadata with
  native TVM reflection, initial stage order, and lowered TIR structural equality.

Ordinary tests and the portable exporter need only Python 3.11. Regenerating
the reference artifacts requires the optional historical TVM environment.
The fixture programs cover matmul, add/multiply with identical names and shapes,
padded fused convolution, softmax, reshape/reindexing, shared branches, multiple
outputs, mixed precision, nonzero reduction minima, scalar constants, Unicode
names, and tuple reducers.

Native validation used TVM built without LLVM/CUDA, Python 3.8.20, NumPy 1.23.5,
and CPU PyTorch 1.13.1. No tensor schedules were compiled for execution and no
program performance was measured. The complete expression graphs were compared
by native reflection, then lowered independently and checked using TVM's own
structural equality implementation. Direct TE tensor equality is not a valid
roundtrip oracle in this fork because producer tensors compare by pointer identity.

This validates the reader against native serialization. The user's full CPU
v3.3 archive still needs conversion on their own machine; unsupported registry
globals and expression nodes fail explicitly.
