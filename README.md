# Canvas Threshold Finder (1+1D)

Self-adjusting binary search that finds the critical threshold R where bound states form in the canvas model. Uses locked weights derived from first principles.

## What It Verifies

The canvas model predicts that spacetime voxels form when R = d+1 = 4 is exceeded, corresponding to N = e^4 ≈ 55 e-folds of inflation. This program searches for the critical R value and verifies bound state formation at the predicted value.

## Quick Start

```

pip install numpy matplotlib scipy
python canvas_threshold_finder.py

```

## How It Works

1. Coarse scan tests R values from 0.5 to 10.0
2. Binary search narrows to the critical R where formation transitions
3. Verification runs multiple trials at R = 4.0
4. Auto-adjusts wave packet amplitude if no bound state detected

## Results

- Critical R found via binary search
- Verification at predicted R = 4.0
- Locked weights: c_eff/d_eff = π/2

## Citation

Ong, E. A Unified Framework of Fundamental Physics (2026).

## License

MIT
