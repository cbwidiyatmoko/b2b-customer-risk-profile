# Optional RW-UCFI Perturbation Test - Appendix / Future Work

Purpose:
Evaluate whether features prioritized by RW-UCFI produce stronger changes in model output when perturbed.

Recommended design:
1. Select Top-K features from Full RW-UCFI and SHAP-only rankings.
2. For each selected feature, replace its value with a neutral baseline, such as the median value from the SHAP explanation sample.
3. Recompute predicted probabilities using the trained model.
4. Measure output change using:
   - absolute change in High Risk probability;
   - absolute change in predicted-class probability;
   - change in max probability and margin;
   - change in predicted class if any.
5. Compare the perturbation effect of RW-UCFI Top-K features against SHAP-only Top-K features.

Interpretation:
A stronger average probability change for RW-UCFI-selected features would support the functional relevance of RW-UCFI prioritization. If the effect is weak or unstable, the result should be reported as a limitation and future validation direction.

Status in this notebook:
By default, perturbation is not executed because it can be computationally expensive and requires access to the trained model plus the exact SHAP explanation sample.