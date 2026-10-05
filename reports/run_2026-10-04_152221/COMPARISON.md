# Comparison across the sweep

27 configuration(s). Axes varying in this sweep: **m, target**.

Three innovation protocols are used and they must not be confused.

* **shared** - target and learned recursion driven by the SAME stream. This is
  the synchronous coupling; `sup|X-Xhat|` is what the comparison lemma bounds,
  and it is an upper bound for the adapted Wasserstein distance, never a lower
  one.
* **independent** - fresh, unrelated streams on both sides. Everything under
  "the law" uses this, so the marginal and the autocorrelation are genuine
  law-level comparisons and not artefacts of a shared driver.
* **matched Gaussian AR** - an AR(m) fitted to the same training path and
  simulated on its own stream. The null: it reproduces the linear
  autocorrelation by construction, so anything the generator gets right that it
  does not is a non-Gaussian, non-linear feature.

`theta` is a maximum over a grid of 64 levels and 3000 windows, reported on the
data support and on the whole box. The bound uses the data-support value; the
proposition as usually stated asks for the box. The two differ by a factor of
roughly 2 to 8, and the gap is not an artefact: the recursion lives on a
forward-invariant set strictly inside the box, which is what `inv_R` reports.


## 1. Hypotheses

`S_m` and `L_(m+1)` are properties of the TARGET at the fitting arity m; `Lip` is the same quantity measured on the FIT. A fit is not told S.

| run | m | S req | S_m | L(m+1) | ΣLip box | ΣLip data | θ data | θ box | bound |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_m02 | 2 | 0.750 | 0.7500 | 0.0000 | 0.9906 | 0.8481 | 0.0458 | 0.2105 | 0.183 |
| linear_m04 | 4 | 0.750 | 0.7500 | 0.0000 | 1.0457 | 0.8269 | 0.0247 | 0.1820 | 0.099 |
| linear_m08 | 8 | 0.750 | 0.7500 | 0.0000 | 1.0815 | 0.8203 | 0.0242 | 0.1300 | 0.097 |
| hetero_m02 | 2 | 0.750 | 0.7500 | 0.0000 | 0.9807 | 0.8067 | 0.0319 | 0.2517 | 0.128 |
| hetero_m04 | 4 | 0.750 | 0.7500 | 0.0000 | 1.0814 | 0.8218 | 0.0239 | 0.1850 | 0.095 |
| hetero_m08 | 8 | 0.750 | 0.7500 | 0.0000 | 1.1427 | 0.8239 | 0.0270 | 0.1598 | 0.108 |
| skew_m02 | 2 | 0.750 | 0.7500 | 0.0000 | 0.9094 | 0.7830 | 0.0344 | 0.1883 | 0.138 |
| skew_m04 | 4 | 0.750 | 0.7500 | 0.0000 | 1.0544 | 0.8205 | 0.0345 | 0.1309 | 0.138 |
| skew_m08 | 8 | 0.750 | 0.7500 | 0.0000 | 1.0585 | 0.7930 | 0.0342 | 0.1195 | 0.137 |
| smoothnl_m02 | 2 | 0.750 | 0.7500 | 0.0000 | 0.9311 | 0.8090 | 0.0700 | 0.1584 | 0.280 |
| smoothnl_m04 | 4 | 0.750 | 0.7500 | 0.0000 | 0.9888 | 0.8292 | 0.0665 | 0.1175 | 0.266 |
| smoothnl_m08 | 8 | 0.750 | 0.7500 | 0.0000 | 1.0058 | 0.8477 | 0.0982 | 0.2238 | 0.393 |
| arch_m02 | 2 | 0.750 | 0.7500 | 0.0000 | 0.5379 | 0.4637 | 0.1646 | 0.3095 | 0.658 |
| arch_m04 | 4 | 0.750 | 0.7500 | 0.0000 | 0.7397 | 0.5341 | 0.1716 | 0.3907 | 0.686 |
| arch_m08 | 8 | 0.750 | 0.7500 | 0.0000 | 0.6845 | 0.4828 | 0.1850 | 0.3656 | 0.740 |
| oscillatory_m02 | 2 | 0.750 | 0.7500 | 0.0000 | 0.8177 | 0.7979 | 0.0957 | 0.2781 | 0.383 |
| oscillatory_m04 | 4 | 0.750 | 0.7500 | 0.0000 | 1.1029 | 0.9405 | 0.0838 | 0.2570 | 0.335 |
| oscillatory_m08 | 8 | 0.750 | 0.7500 | 0.0000 | 1.1123 | 0.9047 | 0.0824 | 0.2659 | 0.329 |
| logistic_m02 | 2 | 0.750 | 0.7500 | 0.0000 | 0.7890 | 0.7616 | 0.1545 | 0.2511 | 0.618 |
| logistic_m04 | 4 | 0.750 | 0.7500 | 0.0000 | 0.8879 | 0.7297 | 0.1763 | 0.2924 | 0.705 |
| logistic_m08 | 8 | 0.750 | 0.7500 | 0.0000 | 0.7070 | 0.6659 | 0.1830 | 0.2953 | 0.732 |
| arma_m02 | 2 | 0.750 | 0.6851 | 0.0649 | 0.8642 | 0.7639 | 0.0247 | 0.4378 | 0.490 |
| arma_m04 | 4 | 0.750 | 0.7444 | 0.0056 | 1.0260 | 0.8216 | 0.0229 | 0.2379 | 0.134 |
| arma_m08 | 8 | 0.750 | 0.7500 | 0.0000 | 1.1665 | 0.8238 | 0.0250 | 0.1207 | 0.100 |
| longmem_m02 | 2 | 0.750 | 0.3934 | 0.3566 | 0.7864 | 0.6009 | 0.0405 | 0.2522 | 1.243 |
| longmem_m04 | 4 | 0.750 | 0.4856 | 0.2644 | 0.9321 | 0.6735 | 0.0328 | 0.1653 | 1.092 |
| longmem_m08 | 8 | 0.750 | 0.5599 | 0.1901 | 0.9451 | 0.6937 | 0.0259 | 0.1159 | 0.923 |


## 2. Repair conditions

Crossings are measured on the DATA SUPPORT; the minimum of dq̂/du and sup|q̂| on the whole BOX. A fit can be monotone where the process lives and not monotone in a corner.

| run | crossing frac | min ∂q̂/∂u (box) | sup\|q̂\| (box) | X-valued | inv. R | repairs needed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_m02 | 0.00e+00 | +0.0520 | 0.862 | yes | 0.80 | none |
| linear_m04 | 0.00e+00 | +0.1752 | 0.933 | yes | 0.85 | none |
| linear_m08 | 0.00e+00 | +0.2014 | 0.926 | yes | 0.90 | none |
| hetero_m02 | 0.00e+00 | -0.0916 | 0.696 | yes | 0.60 | R (in the box only) |
| hetero_m04 | 0.00e+00 | +0.0352 | 0.800 | yes | 0.60 | none |
| hetero_m08 | 0.00e+00 | +0.0760 | 0.819 | yes | 0.60 | none |
| skew_m02 | 0.00e+00 | +0.1503 | 0.812 | yes | 0.75 | none |
| skew_m04 | 0.00e+00 | +0.3239 | 0.916 | yes | 0.80 | none |
| skew_m08 | 0.00e+00 | +0.3819 | 0.925 | yes | 0.85 | none |
| smoothnl_m02 | 0.00e+00 | +0.3541 | 0.887 | yes | 0.90 | none |
| smoothnl_m04 | 0.00e+00 | +0.4486 | 0.934 | yes | 0.90 | none |
| smoothnl_m08 | 0.00e+00 | +0.4246 | 1.023 | no | 1.10 | Pi |
| arch_m02 | 0.00e+00 | +0.6581 | 0.784 | yes | 0.75 | none |
| arch_m04 | 0.00e+00 | +0.6758 | 0.812 | yes | 0.75 | none |
| arch_m08 | 0.00e+00 | +0.6859 | 0.796 | yes | 0.75 | none |
| oscillatory_m02 | 0.00e+00 | +0.5346 | 0.952 | yes | 0.95 | none |
| oscillatory_m04 | 0.00e+00 | +0.4982 | 0.965 | yes | 0.90 | none |
| oscillatory_m08 | 0.00e+00 | +0.5743 | 1.079 | no | 1.20 | Pi |
| logistic_m02 | 0.00e+00 | +0.8381 | 0.945 | yes | 0.95 | none |
| logistic_m04 | 0.00e+00 | +0.7903 | 0.973 | yes | 1.00 | none |
| logistic_m08 | 0.00e+00 | +0.8560 | 0.965 | yes | 1.00 | none |
| arma_m02 | 0.00e+00 | -0.0511 | 0.698 | yes | 0.60 | R (in the box only) |
| arma_m04 | 0.00e+00 | +0.1284 | 0.865 | yes | 0.75 | none |
| arma_m08 | 0.00e+00 | +0.1939 | 0.918 | yes | 0.80 | none |
| longmem_m02 | 0.00e+00 | -0.0070 | 0.655 | yes | 0.60 | R (in the box only) |
| longmem_m04 | 0.00e+00 | +0.0992 | 0.673 | yes | 0.60 | none |
| longmem_m08 | 0.00e+00 | +0.2031 | 0.677 | yes | 0.60 | none |


## 3. Pathwise, shared innovations

| run | sup\|X−X̂\| | mean | bound | slack | holds | spread end | E log Σ\|∂q̂/∂z\| |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_m02 | 0.0240 | 0.0042 | 0.183 | ×7.6 | True | 4.1e-08 | -0.283 |
| linear_m04 | 0.0238 | 0.0045 | 0.099 | ×4.1 | True | 2.8e-08 | -0.246 |
| linear_m08 | 0.0282 | 0.0052 | 0.097 | ×3.4 | True | 1.3e-08 | -0.237 |
| hetero_m02 | 0.0183 | 0.0026 | 0.128 | ×7.0 | True | 6.9e-14 | -0.495 |
| hetero_m04 | 0.0168 | 0.0030 | 0.095 | ×5.7 | True | 1.3e-14 | -0.455 |
| hetero_m08 | 0.0231 | 0.0035 | 0.108 | ×4.7 | True | 1.3e-16 | -0.437 |
| skew_m02 | 0.0326 | 0.0032 | 0.138 | ×4.2 | True | 2.0e-10 | -0.373 |
| skew_m04 | 0.0343 | 0.0045 | 0.138 | ×4.0 | True | 1.8e-10 | -0.339 |
| skew_m08 | 0.0327 | 0.0059 | 0.137 | ×4.2 | True | 1.4e-11 | -0.322 |
| smoothnl_m02 | 0.0475 | 0.0072 | 0.280 | ×5.9 | True | 2.2e-09 | -0.371 |
| smoothnl_m04 | 0.0534 | 0.0080 | 0.266 | ×5.0 | True | 5.6e-10 | -0.328 |
| smoothnl_m08 | 0.0994 | 0.0106 | 0.393 | ×4.0 | True | 3.0e-09 | -0.305 |
| arch_m02 | 0.0983 | 0.0088 | 0.658 | ×6.7 | True | 0.0e+00 | -2.820 |
| arch_m04 | 0.1151 | 0.0104 | 0.686 | ×6.0 | True | 5.6e-17 | -2.508 |
| arch_m08 | 0.1437 | 0.0121 | 0.740 | ×5.1 | True | 1.4e-16 | -2.543 |
| oscillatory_m02 | 0.0814 | 0.0082 | 0.383 | ×4.7 | True | 0.0e+00 | -0.736 |
| oscillatory_m04 | 0.0918 | 0.0099 | 0.335 | ×3.7 | True | 0.0e+00 | -0.808 |
| oscillatory_m08 | 0.0877 | 0.0113 | 0.329 | ×3.8 | True | 1.1e-16 | -0.653 |
| logistic_m02 | 0.0933 | 0.0116 | 0.618 | ×6.6 | True | 0.0e+00 | -1.667 |
| logistic_m04 | 0.1214 | 0.0141 | 0.705 | ×5.8 | True | 0.0e+00 | -1.532 |
| logistic_m08 | 0.1520 | 0.0165 | 0.732 | ×4.8 | True | 1.7e-16 | -1.361 |
| arma_m02 | 0.0263 | 0.0044 | 0.490 | ×18.7 | True | 1.4e-16 | -0.430 |
| arma_m04 | 0.0202 | 0.0034 | 0.134 | ×6.6 | True | 0.0e+00 | -0.290 |
| arma_m08 | 0.0229 | 0.0037 | 0.100 | ×4.4 | True | 4.2e-17 | -0.249 |
| longmem_m02 | 0.0939 | 0.0190 | 1.243 | ×13.2 | True | 5.6e-17 | -0.756 |
| longmem_m04 | 0.0640 | 0.0143 | 1.092 | ×17.0 | True | 6.9e-12 | -0.610 |
| longmem_m08 | 0.0456 | 0.0103 | 0.923 | ×20.2 | True | 3.1e-07 | -0.519 |


## 4. The two laws, independent streams

| run | W₁/sd | W₁/sd (AR) | cond W₁ | cond W₁ (AR) | sd ratio | ACF₁ tru/gen | ACFsq₁ tru/gen/AR | q₀.₉₉₉ err/sd | MMD p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_m02 | 0.0269 | 0.0140 | 0.0042 | 0.0058 | 1.031 | +0.676 / +0.682 | +0.478 / +0.475 / +0.459 | 0.024 | 0.854 |
| linear_m04 | 0.0268 | 0.0140 | 0.0041 | 0.0058 | 1.031 | +0.676 / +0.684 | +0.478 / +0.479 / +0.460 | 0.052 | 0.841 |
| linear_m08 | 0.0277 | 0.0140 | 0.0043 | 0.0056 | 1.031 | +0.676 / +0.682 | +0.478 / +0.476 / +0.460 | 0.030 | 0.755 |
| hetero_m02 | 0.0243 | 0.0350 | 0.0034 | 0.0062 | 1.030 | +0.519 / +0.528 | +0.310 / +0.311 / +0.274 | -0.009 | 0.854 |
| hetero_m04 | 0.0232 | 0.0350 | 0.0035 | 0.0062 | 1.028 | +0.519 / +0.529 | +0.310 / +0.312 / +0.274 | -0.018 | 0.861 |
| hetero_m08 | 0.0262 | 0.0349 | 0.0038 | 0.0062 | 1.033 | +0.519 / +0.527 | +0.310 / +0.309 / +0.274 | -0.019 | 0.808 |
| skew_m02 | 0.0172 | 0.0223 | 0.0049 | 0.0191 | 1.017 | +0.600 / +0.610 | +0.397 / +0.395 / +0.366 | 0.019 | 0.821 |
| skew_m04 | 0.0135 | 0.0222 | 0.0050 | 0.0191 | 1.013 | +0.600 / +0.610 | +0.397 / +0.404 / +0.366 | 0.041 | 0.834 |
| skew_m08 | 0.0179 | 0.0222 | 0.0057 | 0.0191 | 1.018 | +0.600 / +0.610 | +0.397 / +0.405 / +0.366 | 0.037 | 0.801 |
| smoothnl_m02 | 0.0221 | 0.0224 | 0.0084 | 0.0107 | 1.026 | +0.519 / +0.525 | +0.254 / +0.256 / +0.272 | -0.059 | 0.907 |
| smoothnl_m04 | 0.0240 | 0.0224 | 0.0082 | 0.0104 | 1.026 | +0.519 / +0.529 | +0.254 / +0.252 / +0.272 | -0.074 | 0.868 |
| smoothnl_m08 | 0.0259 | 0.0224 | 0.0082 | 0.0104 | 1.028 | +0.519 / +0.526 | +0.254 / +0.270 / +0.272 | -0.081 | 0.815 |
| arch_m02 | 0.0280 | 0.0401 | 0.0134 | 0.0169 | 1.027 | -0.001 / +0.003 | +0.083 / +0.067 / -0.009 | -0.204 | 0.596 |
| arch_m04 | 0.0259 | 0.0401 | 0.0137 | 0.0171 | 1.026 | -0.001 / +0.005 | +0.083 / +0.056 / -0.009 | -0.229 | 0.609 |
| arch_m08 | 0.0324 | 0.0402 | 0.0148 | 0.0169 | 1.028 | -0.001 / +0.000 | +0.083 / +0.041 / -0.009 | -0.284 | 0.543 |
| oscillatory_m02 | 0.0202 | 0.0345 | 0.0108 | 0.0311 | 1.023 | +0.166 / +0.164 | +0.001 / +0.004 / +0.023 | -0.140 | 0.722 |
| oscillatory_m04 | 0.0194 | 0.0345 | 0.0123 | 0.0310 | 1.021 | +0.166 / +0.170 | +0.001 / +0.003 / +0.022 | -0.160 | 0.742 |
| oscillatory_m08 | 0.0225 | 0.0345 | 0.0120 | 0.0310 | 1.026 | +0.166 / +0.165 | +0.001 / +0.029 / +0.022 | -0.157 | 0.722 |
| logistic_m02 | 0.0218 | 0.0398 | 0.0165 | 0.0465 | 1.023 | -0.098 / -0.096 | +0.041 / +0.027 / +0.000 | -0.119 | 0.709 |
| logistic_m04 | 0.0216 | 0.0398 | 0.0164 | 0.0464 | 1.022 | -0.098 / -0.094 | +0.041 / +0.029 / -0.000 | -0.136 | 0.709 |
| logistic_m08 | 0.0264 | 0.0398 | 0.0177 | 0.0461 | 1.024 | -0.098 / -0.097 | +0.041 / +0.023 / -0.000 | -0.156 | 0.662 |
| arma_m02 | 0.0199 | 0.0240 | 0.0048 | 0.0059 | 1.025 | +0.464 / +0.468 | +0.203 / +0.205 / +0.214 | 0.033 | 0.815 |
| arma_m04 | 0.0180 | 0.0241 | 0.0045 | 0.0060 | 1.022 | +0.464 / +0.469 | +0.203 / +0.203 / +0.215 | 0.014 | 0.841 |
| arma_m08 | 0.0231 | 0.0241 | 0.0047 | 0.0060 | 1.027 | +0.464 / +0.467 | +0.203 / +0.199 / +0.215 | -0.008 | 0.755 |
| longmem_m02 | 0.0238 | 0.0271 | 0.0046 | 0.0057 | 1.028 | +0.370 / +0.376 | +0.152 / +0.151 / +0.137 | 0.004 | 0.589 |
| longmem_m04 | 0.0202 | 0.0274 | 0.0047 | 0.0056 | 1.025 | +0.370 / +0.381 | +0.152 / +0.155 / +0.136 | -0.005 | 0.795 |
| longmem_m08 | 0.0257 | 0.0272 | 0.0050 | 0.0060 | 1.030 | +0.370 / +0.379 | +0.152 / +0.151 / +0.135 | -0.009 | 0.682 |


## 5. Early-exercise premium

Bermudan minus European, the functional that needs the adapted structure rather than the path law.

| run | premium target | premium generated | premium AR | relative error |
| :--- | ---: | ---: | ---: | ---: |
| linear_m02 | 0.05884 | 0.05926 | 0.05889 | 0.0071 |
| linear_m04 | 0.05884 | 0.05904 | 0.05883 | 0.0033 |
| linear_m08 | 0.05884 | 0.05898 | 0.05883 | 0.0023 |
| hetero_m02 | 0.05705 | 0.05756 | 0.05608 | 0.0090 |
| hetero_m04 | 0.05705 | 0.05743 | 0.05607 | 0.0066 |
| hetero_m08 | 0.05705 | 0.05768 | 0.05609 | 0.0111 |
| skew_m02 | 0.07716 | 0.07773 | 0.07614 | 0.0074 |
| skew_m04 | 0.07716 | 0.07711 | 0.07613 | -0.0006 |
| skew_m08 | 0.07716 | 0.07757 | 0.07615 | 0.0054 |
| smoothnl_m02 | 0.11798 | 0.11903 | 0.11802 | 0.0090 |
| smoothnl_m04 | 0.11798 | 0.11905 | 0.11804 | 0.0091 |
| smoothnl_m08 | 0.11798 | 0.11895 | 0.11800 | 0.0082 |
| arch_m02 | 0.20539 | 0.20859 | 0.20623 | 0.0155 |
| arch_m04 | 0.20539 | 0.20947 | 0.20621 | 0.0199 |
| arch_m08 | 0.20539 | 0.20957 | 0.20626 | 0.0203 |
| oscillatory_m02 | 0.18135 | 0.18461 | 0.17931 | 0.0179 |
| oscillatory_m04 | 0.18135 | 0.18444 | 0.17918 | 0.0170 |
| oscillatory_m08 | 0.18135 | 0.18370 | 0.17912 | 0.0129 |
| logistic_m02 | 0.21547 | 0.21961 | 0.22097 | 0.0192 |
| logistic_m04 | 0.21547 | 0.21897 | 0.22097 | 0.0163 |
| logistic_m08 | 0.21547 | 0.21974 | 0.22105 | 0.0198 |
| arma_m02 | 0.07116 | 0.07256 | 0.07169 | 0.0197 |
| arma_m04 | 0.07116 | 0.07229 | 0.07169 | 0.0158 |
| arma_m08 | 0.07116 | 0.07268 | 0.07168 | 0.0213 |
| longmem_m02 | 0.07753 | 0.07890 | 0.07687 | 0.0177 |
| longmem_m04 | 0.07753 | 0.07844 | 0.07694 | 0.0117 |
| longmem_m08 | 0.07753 | 0.07858 | 0.07708 | 0.0135 |


## Files

```
<run folder>/results.json   every number
<run folder>/results.txt    the same, laid out
<run folder>/figures.png    nine panels
<run folder>/net.pt         the fitted weights, to re-analyse
                            without retraining
summary.csv                 one flat row per configuration
COMPARISON.md               this file
overview.png                every run side by side
sweep_<axis>.png            one panel set per varying axis
config_used.json            the exact configuration of this run
```
