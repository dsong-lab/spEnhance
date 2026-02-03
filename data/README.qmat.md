# qmat split files

The original `qmat.npz` has been split into 5 files to keep each under ~100MB for GitHub:

- `qmat_part1.npz`
- `qmat_part2.npz`
- `qmat_part3.npz`
- `qmat_part4.npz`
- `qmat_part5.npz`

`rctd_core.load_qmat_npz()` will automatically load these split files if `qmat.npz` is not present.

If you still have the original `qmat.npz`, you can keep it instead of the parts.
