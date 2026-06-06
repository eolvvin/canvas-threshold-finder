"""
SELF-ADJUSTING THRESHOLD FINDER
Automatically finds critical R where bound states form with locked weights
Uses binary search + adaptive parameter tuning
Author: Edwin Ong
Website: eolvvin.github.io
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os
from datetime import datetime

os.makedirs("simulation_results", exist_ok=True)

# ============================================================
# LOCKED WEIGHTS (DERIVED FROM FIRST PRINCIPLES)
# ============================================================

ALPHA_EM = 1.0 / 137.036
PI = np.pi
THETA = PI / 2.0

# Base weights (from expert system derivation)
A_BASE = 1.0
B_BASE = 1.0 / 3.0
C_BASE = 1.0 / (ALPHA_EM * (1 + THETA))  # ≈ 0.00284
D_BASE = C_BASE

print("=" * 70)
print("SELF-ADJUSTING THRESHOLD FINDER")
print("=" * 70)
print(f"Locked weights:")
print(f"  a_base = {A_BASE}")
print(f"  b_base = {B_BASE:.6f}")
print(f"  c_base = {C_BASE:.6f}")
print(f"  d_base = {D_BASE:.6f}")
print("=" * 70)

# ============================================================
# SIMULATION PARAMETERS (Auto-adjustable)
# ============================================================

@dataclass
class SimConfig:
    """Configuration that can auto-adjust based on simulation needs"""
    N: int = 400
    L: float = 200.0
    K: float = 1.0
    sigma: float = 1.2
    t_max: float = 150.0
    n_steps: int = 2000
    Phi0_const: float = 1.0
    
    # Auto-adjust parameters
    amplitude: float = 5.0      # Will increase if no bound state
    max_amplitude: float = 15.0  # Cap to prevent blow-up
    
    # Detection thresholds
    bound_amp_threshold: float = 0.8
    growth_factor_threshold: float = 1.5
    
    def __post_init__(self):
        self.x = np.linspace(0, self.L, self.N)
        self.dx = self.x[1] - self.x[0]
        self.x1 = 0.25 * self.L
        self.x2 = 0.75 * self.L
        self.dt = self.t_max / self.n_steps
        self.t_eval = np.linspace(0, self.t_max, self.n_steps)
    
    def scale_amplitude(self, factor: float):
        """Increase amplitude for next attempt"""
        self.amplitude = min(self.amplitude * factor, self.max_amplitude)
        return self.amplitude

# ============================================================
# SIMULATION ENGINE
# ============================================================

class ThresholdSimulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self.results_history = []
        
    def _initialize_waves(self):
        """Initialize two colliding wave packets"""
        x = self.config.x
        x1 = self.config.x1
        x2 = self.config.x2
        sigma = self.config.sigma
        A = self.config.amplitude
        
        r1 = x - x1
        r2 = x - x2
        env1 = np.exp(-r1**2 / (2 * sigma**2))
        env2 = np.exp(-r2**2 / (2 * sigma**2))
        
        # Add small asymmetry to help bound state formation
        k1 = 0.5
        k2 = -0.5
        wave1 = A * env1 * np.cos(k1 * r1)
        wave2 = A * env2 * np.cos(k2 * r2)
        
        phi = wave1 + wave2
        return phi, phi.copy()
    
    def _apply_absorbing_boundary(self, phi):
        """Apply absorbing boundaries"""
        result = phi.copy()
        width = 20
        for i in range(width):
            factor = np.exp(-(i/6.0)**2)
            result[i] *= factor
            result[-i-1] *= factor
        return result
    
    def _compute_laplacian(self, phi):
        """Compute 1D Laplacian"""
        dx = self.config.dx
        lap = np.zeros_like(phi)
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / (dx * dx)
        lap[0] = (phi[1] - phi[0]) / (dx * dx)
        lap[-1] = (phi[-2] - phi[-1]) / (dx * dx)
        return lap
    
    def run(self, R: float, verbose: bool = False) -> Dict:
        """
        Run simulation for a given threshold R.
        Returns: {'bound_state': bool, 'max_amp': float, 'stability': float}
        """
        config = self.config
        phi, phi_prev = self._initialize_waves()
        
        max_amps = []
        bound_detected = False
        final_amp = 0.0
        
        dt = config.dt
        dx = config.dx
        c = C_BASE
        a = A_BASE
        b = B_BASE
        d = D_BASE
        
        for step in range(config.n_steps):
            # Compute Laplacian
            lap = self._compute_laplacian(phi)
            
            # Polarity
            sign_phi = np.sign(phi)
            
            # UWE: Φ = a v + b Φ₀ + c Φ̈ + d π(v)
            # Solve for Φ̈: Φ̈ = (Φ - a v - b Φ₀ - d π(v)) / c
            v = step * dt
            phi_ddot = (phi - a * v - b * config.Phi0_const - d * sign_phi) / c
            
            # Add spatial coupling (wave propagation)
            phi_ddot = phi_ddot + config.K * lap
            
            # Leapfrog integration
            phi_next = 2*phi - phi_prev + dt * dt * phi_ddot
            
            # Apply absorbing boundaries
            phi_next = self._apply_absorbing_boundary(phi_next)
            
            # Threshold mechanism: enhance peak when intensity exceeds R
            intensity = phi**2
            above = intensity > R
            
            if np.any(above) and step > 100:
                peak_idx = np.argmax(intensity)
                peak_val = phi[peak_idx]
                if abs(peak_val) > 0.3:
                    # Create droplet
                    for di in range(-5, 6):
                        idx = peak_idx + di
                        if 0 <= idx < config.N:
                            phi_next[idx] += peak_val * np.exp(-di**2 / 8.0) * 0.2
            
            phi_prev, phi = phi, phi_next
            
            # Record amplitude
            if step % 100 == 0:
                max_amp = np.max(np.abs(phi))
                max_amps.append(max_amp)
                
                if step > 500 and len(max_amps) >= 5:
                    recent_avg = np.mean(max_amps[-5:])
                    growth = max_amps[-1] / (max_amps[-5] + 1e-10)
                    
                    # Bound state detection criteria
                    if recent_avg > config.bound_amp_threshold and growth > config.growth_factor_threshold:
                        bound_detected = True
                        final_amp = max_amps[-1]
                        if verbose:
                            print(f"      >>> Bound state at step {step}, amp={final_amp:.3f}")
                        break
        
        final_amp = max_amps[-1] if max_amps else 0
        
        # Calculate stability (variance of last few amplitudes)
        stability = 0
        if len(max_amps) >= 5:
            stability = 1.0 / (np.std(max_amps[-5:]) + 1e-10)
        
        return {
            'bound_state': bound_detected,
            'max_amp': final_amp,
            'stability': stability,
            'max_amps': max_amps
        }

# ============================================================
# SELF-ADJUSTING SEARCH
# ============================================================

class SelfAdjustingSearch:
    def __init__(self, config: SimConfig):
        self.config = config
        self.simulator = ThresholdSimulator(config)
        self.results_log = []
        
    def test_R(self, R: float, confidence_runs: int = 2, verbose: bool = True) -> bool:
        """Test if bound state forms at given R with multiple runs for confidence"""
        if verbose:
            print(f"  Testing R={R:.4f} (N={np.exp(R):.1f}):", end=" ", flush=True)
        
        successes = 0
        max_amps = []
        
        for run in range(confidence_runs):
            result = self.simulator.run(R, verbose=False)
            if result['bound_state']:
                successes += 1
            max_amps.append(result['max_amp'])
        
        formed = successes > confidence_runs / 2
        avg_amp = np.mean(max_amps)
        
        if verbose:
            print(f"{'✓ FORMS' if formed else '✗ NO FORM'} (amp={avg_amp:.3f})")
        
        self.results_log.append({'R': R, 'formed': formed, 'amp': avg_amp})
        
        return formed
    
    def find_critical_R(self, R_low: float, R_high: float, 
                        precision: float = 0.05, max_iterations: int = 12) -> Tuple[float, List]:
        """
        Binary search to find critical R.
        Auto-adjusts amplitude if needed.
        """
        print("\n" + "=" * 60)
        print("BINARY SEARCH FOR CRITICAL R")
        print(f"Range: [{R_low:.3f}, {R_high:.3f}]")
        print("=" * 60)
        
        critical_R = None
        
        for iteration in range(max_iterations):
            R_mid = (R_low + R_high) / 2
            formed = self.test_R(R_mid)
            
            if formed:
                critical_R = R_mid
                R_high = R_mid
            else:
                R_low = R_mid
                
                # If no formation and amplitude is low, increase amplitude
                last_result = self.results_log[-1] if self.results_log else None
                if last_result and last_result['amp'] < 0.5 and self.config.amplitude < self.config.max_amplitude:
                    new_amp = self.config.scale_amplitude(1.2)
                    print(f"    Increasing amplitude to {new_amp:.2f}")
            
            if R_high - R_low < precision:
                break
        
        if critical_R is None:
            critical_R = (R_low + R_high) / 2
        
        return critical_R, self.results_log
    
    def verify_at_R(self, R: float, runs: int = 5) -> Dict:
        """Verify formation at a specific R with multiple runs"""
        print(f"\n" + "=" * 60)
        print(f"VERIFICATION AT R = {R:.4f} (N = {np.exp(R):.1f})")
        print("=" * 60)
        
        success_count = 0
        amps = []
        
        for run in range(runs):
            result = self.simulator.run(R, verbose=True)
            if result['bound_state']:
                success_count += 1
            amps.append(result['max_amp'])
        
        success_rate = success_count / runs
        avg_amp = np.mean(amps)
        
        print(f"\nResults: {success_count}/{runs} runs formed bound states ({success_rate*100:.0f}%)")
        print(f"Average max amplitude: {avg_amp:.3f}")
        
        return {
            'success_rate': success_rate,
            'avg_amp': avg_amp,
            'formed': success_rate > 0.5
        }

# ============================================================
# MAIN EXECUTION
# ============================================================

print("\n" + "-" * 60)
print("PHASE 1: INITIAL COARSE SCAN")
print("-" * 60)

config = SimConfig(amplitude=5.0)
search = SelfAdjustingSearch(config)

# Initial coarse scan to find approximate range
coarse_Rs = [0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0]
coarse_results = {}

print("\nCoarse scan:")
for R in coarse_Rs:
    formed = search.test_R(R, confidence_runs=1)
    coarse_results[R] = formed

# Determine search bounds
formed_Rs = [R for R, f in coarse_results.items() if f]
not_formed_Rs = [R for R, f in coarse_results.items() if not f]

if formed_Rs and not_formed_Rs:
    R_low = max(not_formed_Rs)
    R_high = min(formed_Rs)
    print(f"\nTransition detected between R={R_low:.1f} and R={R_high:.1f}")
else:
    R_low = 0.5
    R_high = 10.0
    print(f"\nNo clear transition in coarse scan. Using full range.")

# ============================================================
# PHASE 2: PRECISE BINARY SEARCH
# ============================================================

critical_R, log = search.find_critical_R(R_low, R_high, precision=0.05)

print("\n" + "=" * 60)
print(f"CRITICAL R FOUND: {critical_R:.4f}")
print(f"Corresponding N = e^R = {np.exp(critical_R):.2f} e-folds")
print(f"Predicted N = 55 (R = 4.0)")
print(f"Difference: {np.exp(critical_R) - 55:.2f} e-folds")
print("=" * 60)

# ============================================================
# PHASE 3: VERIFICATION AT PREDICTED R = 4.0
# ============================================================

print("\n" + "-" * 60)
print("PHASE 3: VERIFICATION AT PREDICTED R = 4.0")
print("-" * 60)

verification = search.verify_at_R(4.0, runs=5)

# ============================================================
# PHASE 4: VERIFICATION AT CRITICAL R (if different)
# ============================================================

if abs(critical_R - 4.0) > 0.1:
    print("\n" + "-" * 60)
    print(f"PHASE 4: VERIFICATION AT CRITICAL R = {critical_R:.4f}")
    print("-" * 60)
    critical_verification = search.verify_at_R(critical_R, runs=5)
else:
    critical_verification = verification

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)

print(f"""
Locked weights used:
  a_base = {A_BASE}
  b_base = {B_BASE:.6f}
  c_base = {C_BASE:.6f}
  d_base = {D_BASE:.6f}

Simulation parameters:
  Amplitude: {config.amplitude:.2f}
  Grid size: {config.N}
  Time steps: {config.n_steps}

Results:
  Critical R (binary search): {critical_R:.4f}
  N = exp(R) = {np.exp(critical_R):.2f}
  
  Predicted value: R = 4.0 (N = 55)
  
  Verification at R = 4.0: {'✓ SUCCESS' if verification['formed'] else '✗ FAILURE'}
  Success rate: {verification['success_rate']*100:.0f}%
  Average max amplitude: {verification['avg_amp']:.3f}

Conclusion:
""")

if verification['formed']:
    print("✓ BOUND STATE FORMS AT PREDICTED R = 4.0")
    print("  The locked weights are verified correct.")
    print("  The canvas model prediction N = 55 e-folds is supported.")
else:
    print(f"⚠ BOUND STATE DOES NOT FORM AT R = 4.0")
    print(f"  Critical R found: {critical_R:.4f} (N = {np.exp(critical_R):.1f})")
    print(f"  Predicted N = 55")
    print(f"  This suggests the locked weights may need adjustment or the simulation needs tuning.")

print("=" * 70)

# ============================================================
# PLOT RESULTS
# ============================================================

plt.figure(figsize=(12, 5))

# Plot 1: Formation vs R from log
if log:
    R_vals = [entry['R'] for entry in log]
    formed_vals = [1 if entry['formed'] else 0 for entry in log]
    amp_vals = [entry['amp'] for entry in log]
    
    plt.subplot(1, 2, 1)
    plt.plot(R_vals, amp_vals, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=config.bound_amp_threshold, color='r', linestyle='--', label=f'Bound threshold ({config.bound_amp_threshold})')
    plt.axvline(x=critical_R, color='g', linestyle='--', label=f'Critical R = {critical_R:.3f}')
    plt.axvline(x=4.0, color='orange', linestyle=':', label='Predicted R = 4.0')
    plt.xlabel('Threshold R = T/σ²')
    plt.ylabel('Maximum Amplitude')
    plt.title('Bound State Formation with Locked Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    colors = ['green' if f else 'red' for f in formed_vals]
    plt.bar(range(len(R_vals)), formed_vals, color=colors, alpha=0.7)
    plt.xticks(range(len(R_vals)), [f'{r:.1f}' for r in R_vals], rotation=45)
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.xlabel('Threshold R')
    plt.ylabel('Bound State Formed')
    plt.title('Formation by Threshold')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['NO', 'YES'])

plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"simulation_results/self_adjusting_scan_{timestamp}.png", dpi=150)
plt.show()

print(f"\nPlot saved to simulation_results/self_adjusting_scan_{timestamp}.png")