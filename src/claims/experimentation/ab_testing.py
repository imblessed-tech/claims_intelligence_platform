import math
from dataclasses import dataclass
from typing import Any
import numpy as np
import scipy.stats

@dataclass
class ABTestResult:
    control_name: str
    treatment_name: str
    metric: str
    control_mean: float
    treatment_mean: float
    p_value: float
    effect_size: float
    is_significant: bool
    winner: str
    recommendation: str

class ABTestFramework:
    def run_test(
        self,
        control_scores: Any,
        treatment_scores: Any,
        control_name: str,
        treatment_name: str,
        metric: str,
        alpha: float = 0.05
    ) -> ABTestResult:
        """Run an independent two-sample t-test to compare two model versions."""
        control = np.array(control_scores)
        treatment = np.array(treatment_scores)
        
        mean_control = float(np.mean(control))
        mean_treatment = float(np.mean(treatment))
        
        # Two-sample t-test (independent)
        _, p_value = scipy.stats.ttest_ind(control, treatment)
        p_value = float(p_value)
        
        # Cohen's d: (mean_treatment - mean_control) / pooled_std
        std_control = float(np.std(control, ddof=1))
        std_treatment = float(np.std(treatment, ddof=1))
        pooled_std = math.sqrt((std_control**2 + std_treatment**2) / 2.0)
        
        if pooled_std > 0:
            effect_size = (mean_treatment - mean_control) / pooled_std
        else:
            effect_size = 0.0
            
        is_significant = p_value < alpha
        
        # treatment is the winner if it showed significant improvement
        if is_significant and mean_treatment > mean_control:
            winner = treatment_name
        else:
            winner = control_name
            
        if is_significant:
            recommendation = (
                f"Statistically significant difference detected in {metric} (p = {p_value:.4f}). "
                f"Treatment '{treatment_name}' (mean: {mean_treatment:.4f}) outperforms "
                f"Control '{control_name}' (mean: {mean_control:.4f}) with an effect size of {effect_size:.4f}. "
                f"Recommendation: Deploy {winner}."
            )
        else:
            recommendation = (
                f"No statistically significant difference detected in {metric} (p = {p_value:.4f}). "
                f"Treatment '{treatment_name}' (mean: {mean_treatment:.4f}) vs "
                f"Control '{control_name}' (mean: {mean_control:.4f}). Effect size: {effect_size:.4f}. "
                f"Recommendation: Keep {control_name} (Control)."
            )
            
        return ABTestResult(
            control_name=control_name,
            treatment_name=treatment_name,
            metric=metric,
            control_mean=mean_control,
            treatment_mean=mean_treatment,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            winner=winner,
            recommendation=recommendation
        )

    def power_analysis(self, effect_size: float, alpha: float = 0.05, power: float = 0.80) -> int:
        """Calculate the minimum sample size needed to detect an effect."""
        if effect_size == 0:
            return 0
        z_alpha = scipy.stats.norm.ppf(1 - alpha / 2)
        z_power = scipy.stats.norm.ppf(power)
        n = ((z_alpha + z_power) / effect_size) ** 2
        return int(math.ceil(n))

# Compare GBM vs Ridge regression using cross-validation scores
# framework = ABTestFramework()
# result = framework.run_test(
#     gbm_cv_scores, ridge_cv_scores,
#     "GradientBoosting", "Ridge", "r2"
# )
# print(result.recommendation)
