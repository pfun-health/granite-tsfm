#!/usr/bin/env python3
"""
Hybrid TTM + PFun CMA Model Implementation
Strategy 1: Residual Correction (Recommended)

This implementation combines:
- PFun CMA: Mechanistic circadian glucose model
- IBM TTM: Time series foundation model
- Residual learning: TTM corrects CMA baseline predictions

Author: Implementation based on research combining physics-informed 
        neural networks with foundation models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PFun CMA Model Integration
# =============================================================================

try:
    from pfun_cma_model.engine.cma import CMASleepWakeModel
    from pfun_cma_model.engine.fit import fit_model, estimate_mealtimes
    from pfun_cma_model.engine.data_utils import format_data
    PFUN_AVAILABLE = True
except ImportError:
    print("Warning: pfun_cma_model not installed. Install with: pip install pfun-cma-model")
    PFUN_AVAILABLE = False

# =============================================================================
# TTM Model Integration (Hugging Face)
# =============================================================================

try:
    from transformers import AutoModel
    import torch
    TTM_AVAILABLE = True
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers torch")
    TTM_AVAILABLE = False


class HybridTTMCMAModel:
    """
    Hybrid glucose forecasting model combining:
    1. PFun CMA: Physiological baseline (circadian-regulated glucose dynamics)
    2. IBM TTM: Neural correction (learns individual-specific patterns)
    
    Equation: G_pred(t) = G_CMA(t) + G_TTM_correction(t)
    """
    
    def __init__(
        self,
        ttm_model_id: str = "ibm-granite/granite-timeseries-ttm-r2",
        context_length: int = 512,
        forecast_length: int = 96,
        device: str = "cpu"
    ):
        """
        Initialize hybrid model
        
        Args:
            ttm_model_id: Hugging Face model ID for TTM
            context_length: Number of historical points for TTM (512, 1024, or 1536)
            forecast_length: Prediction horizon (typically 96 for 8 hours ahead)
            device: 'cpu' or 'cuda'
        """
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.device = device
        
        # Initialize CMA model (will be fitted to data)
        self.cma_model: Optional[CMASleepWakeModel] = None
        self.cma_params: Optional[Dict] = None
        
        # Initialize TTM model
        if TTM_AVAILABLE:
            try:
                self.ttm_model = AutoModel.from_pretrained(
                    ttm_model_id,
                    trust_remote_code=True
                )
                self.ttm_model.to(device)
                self.ttm_model.eval()
            except Exception as e:
                print(f"Warning: Could not load TTM model: {e}")
                self.ttm_model = None
        else:
            self.ttm_model = None
    
    def fit_cma(
        self,
        glucose_data: pd.DataFrame,
        time_col: str = 'time_decimal',
        glucose_col: str = 'glucose_mgdl',
        meal_times: Optional[np.ndarray] = None,
        estimate_meals: bool = True
    ) -> Dict:
        """
        Fit CMA model to glucose data
        
        Args:
            glucose_data: DataFrame with time and glucose columns
            time_col: Name of time column (decimal hours 0-24)
            glucose_col: Name of glucose column (mg/dL)
            meal_times: Pre-specified meal times (hours), or None to estimate
            estimate_meals: Whether to estimate meal times from data
            
        Returns:
            Dictionary with fitted parameters and metrics
        """
        if not PFUN_AVAILABLE:
            raise RuntimeError("pfun_cma_model not available")
        
        # Prepare data
        glucose_data['ts_utc'] = glucose_data[time_col]
        data_formatted = format_data(
            glucose_data[[time_col, glucose_col]].rename(
                columns={time_col: 'ts_local', glucose_col: 'G'}
            ),
            N=len(glucose_data)
        )
        
        # Estimate meal times if not provided
        if meal_times is None and estimate_meals:
            meal_times = estimate_mealtimes(
                data_formatted,
                ycol='G',
                tm_freq='2h',
                n_meals=3
            )
            print(f"Estimated meal times: {meal_times}")
        
        # Fit CMA model
        fit_result = fit_model(
            data=data_formatted,
            tcol='t',
            ycol='G',
            tM=meal_times,
            N=min(1024, len(glucose_data)),
            curve_fit_kwds={
                'method': 'L-BFGS-B',
                'ftol': 1e-6,
                'xtol': 1e-6,
                'max_nfev': 500000,
                'verbose': 0
            }
        )
        
        # Store fitted model and parameters
        self.cma_model = fit_result.cma
        self.cma_params = fit_result.popt_named
        
        # Compute fit quality metrics
        cma_predictions = fit_result.soln['G'].values
        true_glucose = data_formatted['G'].values
        
        rmse = np.sqrt(np.mean((cma_predictions - true_glucose) ** 2))
        mae = np.mean(np.abs(cma_predictions - true_glucose))
        
        results = {
            'parameters': self.cma_params,
            'rmse': rmse,
            'mae': mae,
            'meal_times': meal_times,
            'fit_quality': 'good' if rmse < 30 else 'fair' if rmse < 40 else 'poor'
        }
        
        print(f"\nCMA Model Fit Results:")
        print(f"  RMSE: {rmse:.2f} mg/dL")
        print(f"  MAE: {mae:.2f} mg/dL")
        print(f"  Fit Quality: {results['fit_quality']}")
        print(f"\nFitted Parameters:")
        for param, value in self.cma_params.items():
            print(f"  {param}: {value:.6f}")
        
        return results
    
    def get_cma_predictions(
        self,
        time_vector: np.ndarray,
        return_components: bool = False
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """
        Generate CMA baseline predictions for given time points
        
        Args:
            time_vector: Time points in decimal hours (0-24)
            return_components: If True, return intermediate physiological signals
            
        Returns:
            CMA glucose predictions, or dict with all components
        """
        if self.cma_model is None:
            raise RuntimeError("CMA model not fitted. Call fit_cma() first.")
        
        # Update CMA model with new time vector
        cma_temp = self.cma_model.update(t=time_vector, inplace=False)
        cma_df = cma_temp.run()
        
        if return_components:
            return {
                'glucose': cma_df['G'].values,
                'light': cma_df['L'].values,
                'melatonin': cma_df['m'].values,
                'cortisol': cma_df['c'].values,
                'adiponectin': cma_df['a'].values,
                'insulin_sensitivity': cma_df['I_S'].values,
                'effective_insulin': cma_df['I_E'].values
            }
        
        return cma_df['G'].values
    
    def prepare_ttm_input(
        self,
        historical_residuals: np.ndarray,
        normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare residuals for TTM input
        
        Args:
            historical_residuals: Past residuals (true_glucose - cma_baseline)
            normalize: Whether to normalize (TTM expects normalized input)
            
        Returns:
            Dictionary with TTM input tensors and normalization stats
        """
        # Ensure correct shape
        if len(historical_residuals) < self.context_length:
            # Pad with zeros if insufficient history
            pad_length = self.context_length - len(historical_residuals)
            historical_residuals = np.concatenate([
                np.zeros(pad_length),
                historical_residuals
            ])
        else:
            # Take last context_length points
            historical_residuals = historical_residuals[-self.context_length:]
        
        # Reshape to (batch, channels, time)
        residuals_tensor = historical_residuals.reshape(1, 1, -1)
        
        # Normalize
        if normalize:
            mean = residuals_tensor.mean()
            std = residuals_tensor.std() + 1e-8
            residuals_normalized = (residuals_tensor - mean) / std
        else:
            mean, std = 0.0, 1.0
            residuals_normalized = residuals_tensor
        
        return {
            'input': torch.FloatTensor(residuals_normalized).to(self.device),
            'mean': mean,
            'std': std
        }
    
    def train_ttm_on_residuals(
        self,
        train_data: pd.DataFrame,
        time_col: str = 'time_decimal',
        glucose_col: str = 'glucose_mgdl',
        val_data: Optional[pd.DataFrame] = None,
        epochs: int = 20,
        learning_rate: float = 1e-4
    ):
        """
        Train TTM to predict residuals (corrections to CMA baseline)
        
        Args:
            train_data: Training glucose data
            time_col: Time column name
            glucose_col: Glucose column name
            val_data: Optional validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        if self.ttm_model is None:
            raise RuntimeError("TTM model not available")
        
        if self.cma_model is None:
            raise RuntimeError("CMA model not fitted. Call fit_cma() first.")
        
        # Get CMA baseline predictions
        cma_baseline = self.get_cma_predictions(train_data[time_col].values)
        
        # Compute residuals
        residuals = train_data[glucose_col].values - cma_baseline
        
        print(f"\nTraining TTM on residuals:")
        print(f"  Residual statistics:")
        print(f"    Mean: {residuals.mean():.2f} mg/dL")
        print(f"    Std: {residuals.std():.2f} mg/dL")
        print(f"    Range: [{residuals.min():.2f}, {residuals.max():.2f}] mg/dL")
        
        # Create training samples
        # (This is simplified - actual TTM training would use proper data loaders)
        num_samples = len(residuals) - self.context_length - self.forecast_length
        
        if num_samples <= 0:
            raise ValueError(
                f"Insufficient data. Need at least {self.context_length + self.forecast_length} "
                f"samples, got {len(residuals)}"
            )
        
        print(f"\n  Training samples: {num_samples}")
        print(f"  Context length: {self.context_length}")
        print(f"  Forecast length: {self.forecast_length}")
        
        # Note: Actual training implementation would go here
        # This requires access to TTM's training API which varies by implementation
        print("\n  [Training loop would be implemented here]")
        print("  For production, integrate with TTM's official training API")
    
    def forecast(
        self,
        historical_glucose: np.ndarray,
        historical_time: np.ndarray,
        forecast_horizon: int = 96,
        return_components: bool = False
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """
        Generate hybrid forecast: CMA baseline + TTM correction
        
        Args:
            historical_glucose: Past glucose readings (mg/dL)
            historical_time: Past time points (decimal hours)
            forecast_horizon: Number of future points to predict
            return_components: If True, return CMA and TTM components separately
            
        Returns:
            Hybrid predictions, or dict with components
        """
        if self.cma_model is None:
            raise RuntimeError("CMA model not fitted")
        
        # Generate future time vector
        time_step = np.diff(historical_time).mean()
        future_time = historical_time[-1] + time_step * np.arange(1, forecast_horizon + 1)
        future_time = future_time % 24  # Wrap to 0-24 hours
        
        # CMA baseline for future
        cma_baseline_future = self.get_cma_predictions(future_time)
        
        # Get CMA baseline for historical data
        cma_baseline_historical = self.get_cma_predictions(historical_time)
        
        # Compute historical residuals
        historical_residuals = historical_glucose - cma_baseline_historical
        
        # TTM correction (if available)
        if self.ttm_model is not None:
            ttm_input = self.prepare_ttm_input(historical_residuals)
            
            with torch.no_grad():
                # Note: Actual TTM API call depends on implementation
                # This is a placeholder for the forecast method
                try:
                    ttm_output = self.ttm_model.forecast(
                        ttm_input['input'],
                        forecast_length=forecast_horizon
                    )
                    
                    # Denormalize
                    ttm_correction = (
                        ttm_output.cpu().numpy().flatten() * ttm_input['std'] 
                        + ttm_input['mean']
                    )
                except Exception as e:
                    print(f"Warning: TTM forecast failed: {e}")
                    print("Falling back to CMA-only predictions")
                    ttm_correction = np.zeros(forecast_horizon)
        else:
            # No TTM available, use CMA only
            ttm_correction = np.zeros(forecast_horizon)
        
        # Hybrid prediction
        hybrid_forecast = cma_baseline_future + ttm_correction
        
        if return_components:
            return {
                'hybrid': hybrid_forecast,
                'cma_baseline': cma_baseline_future,
                'ttm_correction': ttm_correction
            }
        
        return hybrid_forecast
    
    def generate_clinical_report(self) -> str:
        """
        Generate human-readable clinical interpretation of CMA parameters
        
        Returns:
            Clinical report as formatted string
        """
        if self.cma_params is None:
            return "CMA model not fitted yet."
        
        report_lines = ["=" * 80]
        report_lines.append("PERSONALIZED METABOLIC HEALTH REPORT")
        report_lines.append("Based on PFun CMA Physiological Model")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Time zone offset (d)
        d = self.cma_params['d']
        report_lines.append("1. CIRCADIAN ALIGNMENT")
        if abs(d) < 0.5:
            report_lines.append("   ✓ Your circadian rhythm is well-aligned with your local time zone.")
        elif d < -0.5:
            report_lines.append(
                f"   ⚠ Your circadian phase is advanced by {abs(d):.1f} hours "
                "(you may be a 'morning person')."
            )
        else:
            report_lines.append(
                f"   ⚠ Your circadian phase is delayed by {d:.1f} hours "
                "(you may be an 'evening person')."
            )
        report_lines.append("")
        
        # Photoperiod (taup)
        taup = self.cma_params['taup']
        report_lines.append("2. LIGHT EXPOSURE PATTERN")
        if taup < 1.5:
            report_lines.append("   ✓ Normal light exposure pattern detected.")
        elif taup < 2.5:
            report_lines.append(
                "   ⚠ Extended light exposure detected (possible artificial light at night)."
            )
            report_lines.append("   💡 Consider reducing screen time before bed.")
        else:
            report_lines.append(
                "   ⚠⚠ Very long light exposure (potential circadian disruption)."
            )
            report_lines.append("   💡 Strongly consider light hygiene improvements.")
        report_lines.append("")
        
        # Glucose response (taug)
        taug = self.cma_params['taug']
        report_lines.append("3. GLUCOSE METABOLISM")
        if taug < 0.8:
            report_lines.append("   ✓ Rapid glucose clearance (good insulin sensitivity).")
        elif taug < 1.5:
            report_lines.append("   ✓ Normal glucose response time.")
        else:
            report_lines.append("   ⚠ Slow glucose clearance (possible insulin resistance).")
            report_lines.append("   💡 Consider consulting with healthcare provider.")
        report_lines.append("")
        
        # Bias (B)
        B = self.cma_params['B']
        report_lines.append("4. BASELINE GLUCOSE")
        if B < 0.08:
            report_lines.append("   ✓ Normal fasting glucose baseline.")
        elif B < 0.15:
            report_lines.append("   ⚠ Elevated baseline glucose.")
            report_lines.append("   💡 Monitor for prediabetes risk factors.")
        else:
            report_lines.append("   ⚠⚠ High baseline glucose.")
            report_lines.append("   💡 Consult with endocrinologist for evaluation.")
        report_lines.append("")
        
        # Cortisol sensitivity (Cm)
        Cm = self.cma_params['Cm']
        report_lines.append("5. STRESS RESPONSE")
        if Cm < 0.3:
            report_lines.append("   ✓ Low metabolic stress response.")
        elif Cm < 1.0:
            report_lines.append("   ✓ Normal cortisol-mediated glucose regulation.")
        else:
            report_lines.append(
                "   ⚠ High cortisol sensitivity "
                "(stress may significantly impact glucose)."
            )
            report_lines.append("   💡 Consider stress management techniques.")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("PARAMETERS (for reference):")
        for param, value in self.cma_params.items():
            report_lines.append(f"  {param}: {value:.6f}")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


# =============================================================================
# Example Usage
# =============================================================================
def generate_sythetic_data():
    np.random.seed(42)
    # Generate datetime vector at 5-minute intervals over 24 hours starting from a base date
    from datetime import datetime, timedelta
    base_datetime = datetime(2025, 10, 18, 0, 0, 0)
    interval_minutes = 5
    n_points = 288  # 24*60/5=288
    time_points_dt = [base_datetime + timedelta(minutes=i * interval_minutes) for i in range(n_points)]
    # For compatibility, also keep decimal hours (for circadian simulation, etc.)
    decimal_times = [dt.hour + dt.minute/60.0 + dt.second/3600.0 for dt in time_points_dt]

    # Simulate glucose with circadian pattern + meals
    base_glucose = 100 + 20 * np.sin(2 * np.pi * (np.array(decimal_times) - 4) / 24)
    meal_effects = (
        30 * np.exp(-((np.array(decimal_times) - 7)**2) / 2) +  # Breakfast
        35 * np.exp(-((np.array(decimal_times) - 12)**2) / 2) +  # Lunch
        40 * np.exp(-((np.array(decimal_times) - 18)**2) / 2)    # Dinner
    )
    glucose_data = base_glucose + meal_effects + np.random.normal(0, 5, len(decimal_times))
    sample_data = pd.DataFrame({
        'ts_local': time_points_dt,      # datetime vector for local time
        'time_decimal': decimal_times,   # decimal hours, if needed for compatibility
        'glucose_mgdl': glucose_data
    })
    return sample_data

def example_workflow():
    """
    Example workflow demonstrating hybrid model usage
    """
    print("\n" + "="*80)
    print("HYBRID TTM + PFUN CMA MODEL - EXAMPLE WORKFLOW")
    print("="*80)
    
    # 1. Load sample CGM data
    print("\n[Step 1] Loading sample CGM data...")
    # In practice, load from CSV: pd.read_csv('patient_cgm.csv')
    
    # Synthetic sample data for demonstration
    sample_data = generate_sythetic_data()
    
    print(f"  Loaded {len(sample_data)} data points")
    print(f"  Time range: {sample_data['time_decimal'].min():.1f} to "
          f"{sample_data['time_decimal'].max():.1f} hours")
    print(f"  Glucose range: {sample_data['glucose_mgdl'].min():.1f} to "
          f"{sample_data['glucose_mgdl'].max():.1f} mg/dL")
    
    # 2. Initialize hybrid model
    print("\n[Step 2] Initializing hybrid model...")
    model = HybridTTMCMAModel(
        context_length=512,
        forecast_length=96,
        device='cpu'
    )
    print("  ✓ Model initialized")
    
    # 3. Fit CMA model
    print("\n[Step 3] Fitting CMA physiological model...")
    if PFUN_AVAILABLE:
        cma_fit_results = model.fit_cma(
            sample_data,
            time_col='time_decimal',
            glucose_col='glucose_mgdl',
            estimate_meals=True
        )
        
        # 4. Generate clinical report
        print("\n[Step 4] Generating clinical interpretation...")
        clinical_report = model.generate_clinical_report()
        print(clinical_report)
        
        # 5. Generate predictions
        print("\n[Step 5] Generating hybrid forecasts...")
        # Use first 80% for context, forecast last 20%
        split_idx = int(0.8 * len(sample_data))
        
        historical_glucose = sample_data['glucose_mgdl'].values[:split_idx]
        historical_time = sample_data['time_decimal'].values[:split_idx]
        
        predictions = model.forecast(
            historical_glucose=historical_glucose,
            historical_time=historical_time,
            forecast_horizon=len(sample_data) - split_idx,
            return_components=True
        )
        
        print(f"  Generated {len(predictions['hybrid'])} predictions")
        print(f"  Hybrid mean: {predictions['hybrid'].mean():.1f} mg/dL")
        print(f"  CMA baseline mean: {predictions['cma_baseline'].mean():.1f} mg/dL")
        print(f"  TTM correction mean: {predictions['ttm_correction'].mean():.1f} mg/dL")
        
        # 6. Evaluate predictions
        print("\n[Step 6] Evaluating forecast accuracy...")
        true_glucose = sample_data['glucose_mgdl'].values[split_idx:]
        
        rmse_hybrid = np.sqrt(np.mean((predictions['hybrid'] - true_glucose) ** 2))
        rmse_cma = np.sqrt(np.mean((predictions['cma_baseline'] - true_glucose) ** 2))
        
        print(f"  CMA-only RMSE: {rmse_cma:.2f} mg/dL")
        print(f"  Hybrid RMSE: {rmse_hybrid:.2f} mg/dL")
        
        if rmse_hybrid < rmse_cma:
            improvement = ((rmse_cma - rmse_hybrid) / rmse_cma) * 100
            print(f"  ✓ Hybrid model improved accuracy by {improvement:.1f}%")
        
    else:
        print("  ⚠ pfun_cma_model not available - skipping CMA fitting")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Integrate with real CGM data")
    print("  2. Train TTM component on residuals")
    print("  3. Validate on held-out test data")
    print("  4. Deploy for real-time forecasting")


if __name__ == "__main__":
    example_workflow()
