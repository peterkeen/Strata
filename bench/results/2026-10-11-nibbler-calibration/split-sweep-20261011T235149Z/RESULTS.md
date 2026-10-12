# Completed measurements from a BLOCKED split-sweep window

**Overall FAILED: all eight measurement arms completed, but the journal ended blocked after the SwapFree safety refusal. Raw passed:true is not window success.**

One window, two repeats/arm; installed 0.1.41 GPTQ engine on nibbler 9900X / RTX 5060 Ti. No production recommendation.

## Per-arm aggregate wall rates

Each fresh/warm cell is the arithmetic mean of two group rates. S=178 prompt tokens, L=8243; all outputs 96 tokens. Not pooled token/time throughput.

| Arm / split mode | S1 fresh / warm | S2 fresh / warm | L1 fresh / warm | L2 fresh / warm | Load s | GPU peak / min free MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 00 P0.25 | 31.23 / 55.15 | 34.44 / 45.25 | 11.33 / 55.60 | 10.41 / 43.10 | 27.022 | 14037 / 1853 |
| 01 E0.10 | 31.27 / 54.49 | 34.10 / 44.63 | 11.32 / 55.00 | 10.36 / 42.78 | 26.023 | 14013 / 1877 |
| 02 E0.40 | 31.43 / 55.04 | 33.59 / 45.26 | 11.16 / 54.90 | 10.39 / 43.04 | 26.021 | 14045 / 1845 |
| 03 E0.55 | 29.45 / 49.15 | 31.62 / 40.41 | 10.95 / 48.48 | 10.18 / 39.13 | 25.021 | 14015 / 1875 |
| 04 E0.55 | 28.49 / 47.79 | 31.24 / 40.39 | 10.92 / 47.51 | 10.05 / 38.47 | 27.024 | 14043 / 1847 |
| 05 E0.40 | 30.67 / 54.38 | 33.07 / 44.10 | 11.24 / 54.11 | 10.31 / 41.79 | 29.026 | 14043 / 1847 |
| 06 E0.10 | 30.74 / 53.47 | 33.07 / 44.47 | 11.09 / 53.63 | 10.32 / 42.95 | 27.024 | 14045 / 1845 |
| 07 E0.25 | 31.06 / 54.40 | 34.19 / 45.55 | 11.25 / 55.29 | 10.41 / 42.62 | 27.023 | 14041 / 1849 |

## Per-arm timing and native/API counters

Native/API hit and PCIe share ranges are latest 12 rows, repeat 1 only. API decode range covers the recorded 24 request fields and may describe last native segments.

| Arm | TTFT p95 s | Wall p95 s | API decode tok/s range | Hit % latest12 | PCIe share % latest12 |
| --- | --- | --- | --- | --- | --- |
| 00 | 16.022 | 18.446 | 9.2–63.0 | 37.3–68.0 | 6.1–12.9 |
| 01 | 16.027 | 18.500 | 9.2–63.5 | 31.4–64.6 | 0.2–2.5 |
| 02 | 16.091 | 18.450 | 9.2–61.9 | 29.0–71.7 | 9.8–24.0 |
| 03 | 16.148 | 18.851 | 9.0–56.8 | 47.1–76.6 | 16.2–31.0 |
| 04 | 16.235 | 18.998 | 8.8–54.4 | 47.6–76.4 | 16.2–31.7 |
| 05 | 16.120 | 18.598 | 9.2–62.1 | 39.3–71.6 | 9.6–21.2 |
| 06 | 16.123 | 18.587 | 9.1–64.6 | 25.3–64.4 | 0.0–3.5 |
| 07 | 16.086 | 18.407 | 9.2–63.0 | 37.6–67.9 | 5.8–13.1 |

## Each split versus baseline and observed variation

Difference of arithmetic mean bucket group-wall rates. Within band = larger of baseline and candidate maximum absolute r0-r1 difference within one arm. Full band = larger of baseline and candidate full max-min range across all observed repeats/arms. Absolute mean difference strictly greater than band is descriptive exceedance only, not significance, confidence, or tuning proof.

| Bucket | Split | Mean delta tok/s | Within-arm band tok/s | Exceeds within? | Full observed band tok/s | Also exceeds full? |
| --- | --- | --- | --- | --- | --- | --- |
| 178tok/1stream/fresh | explicit 0.10 | -0.22 | 9.60 | no | 9.63 | no |
| 178tok/1stream/fresh | explicit 0.25 | -0.16 | 7.08 | no | 7.08 | no |
| 178tok/1stream/fresh | explicit 0.40 | -0.18 | 6.94 | no | 7.62 | no |
| 178tok/1stream/fresh | explicit 0.55 | -2.26 | 7.48 | no | 7.98 | no |
| 178tok/1stream/warm | explicit 0.10 | -1.17 | 6.00 | no | 6.73 | no |
| 178tok/1stream/warm | explicit 0.25 | -0.75 | 6.07 | no | 6.07 | no |
| 178tok/1stream/warm | explicit 0.40 | -0.44 | 5.45 | no | 5.45 | no |
| 178tok/1stream/warm | explicit 0.55 | -6.68 | 5.45 | yes | 6.14 | yes |
| 178tok/2stream/fresh | explicit 0.10 | -0.85 | 0.92 | no | 1.57 | no |
| 178tok/2stream/fresh | explicit 0.25 | -0.25 | 0.92 | no | 0.92 | no |
| 178tok/2stream/fresh | explicit 0.40 | -1.11 | 0.92 | yes | 0.92 | yes |
| 178tok/2stream/fresh | explicit 0.55 | -3.00 | 0.92 | yes | 1.06 | yes |
| 178tok/2stream/warm | explicit 0.10 | -0.70 | 0.85 | no | 0.85 | no |
| 178tok/2stream/warm | explicit 0.25 | +0.30 | 0.60 | no | 0.60 | no |
| 178tok/2stream/warm | explicit 0.40 | -0.56 | 0.60 | no | 1.37 | no |
| 178tok/2stream/warm | explicit 0.55 | -4.85 | 0.81 | yes | 0.81 | yes |
| 8243tok/1stream/fresh | explicit 0.10 | -0.13 | 0.15 | no | 0.32 | no |
| 8243tok/1stream/fresh | explicit 0.25 | -0.08 | 0.14 | no | 0.14 | no |
| 8243tok/1stream/fresh | explicit 0.40 | -0.13 | 0.08 | yes | 0.12 | yes |
| 8243tok/1stream/fresh | explicit 0.55 | -0.39 | 0.20 | yes | 0.20 | yes |
| 8243tok/1stream/warm | explicit 0.10 | -1.28 | 1.38 | no | 2.42 | no |
| 8243tok/1stream/warm | explicit 0.25 | -0.31 | 1.57 | no | 1.57 | no |
| 8243tok/1stream/warm | explicit 0.40 | -1.10 | 1.38 | no | 1.86 | no |
| 8243tok/1stream/warm | explicit 0.55 | -7.60 | 1.42 | yes | 1.89 | yes |
| 8243tok/2stream/fresh | explicit 0.10 | -0.07 | 0.05 | yes | 0.06 | yes |
| 8243tok/2stream/fresh | explicit 0.25 | +0.00 | 0.05 | no | 0.05 | no |
| 8243tok/2stream/fresh | explicit 0.40 | -0.06 | 0.03 | yes | 0.12 | no |
| 8243tok/2stream/fresh | explicit 0.55 | -0.29 | 0.12 | yes | 0.20 | yes |
| 8243tok/2stream/warm | explicit 0.10 | -0.24 | 1.05 | no | 1.05 | no |
| 8243tok/2stream/warm | explicit 0.25 | -0.47 | 0.93 | no | 0.93 | no |
| 8243tok/2stream/warm | explicit 0.40 | -0.68 | 0.93 | no | 2.01 | no |
| 8243tok/2stream/warm | explicit 0.55 | -4.30 | 0.93 | yes | 1.18 | yes |

## Repeated explicit-arm drift

| Split | Bucket | Between-arm mean difference tok/s | Largest within-arm r0/r1 difference | Full observed range |
| --- | --- | --- | --- | --- |
| explicit 0.10 | 178tok/1stream/fresh | 0.528 | 9.596 | 9.625 |
| explicit 0.10 | 178tok/1stream/warm | 1.024 | 6.001 | 6.730 |
| explicit 0.10 | 178tok/2stream/fresh | 1.031 | 0.651 | 1.568 |
| explicit 0.10 | 178tok/2stream/warm | 0.150 | 0.850 | 0.850 |
| explicit 0.10 | 8243tok/1stream/fresh | 0.223 | 0.154 | 0.324 |
| explicit 0.10 | 8243tok/1stream/warm | 1.365 | 1.378 | 2.419 |
| explicit 0.10 | 8243tok/2stream/fresh | 0.031 | 0.046 | 0.060 |
| explicit 0.10 | 8243tok/2stream/warm | 0.174 | 1.051 | 1.051 |
| explicit 0.40 | 178tok/1stream/fresh | 0.759 | 6.896 | 7.624 |
| explicit 0.40 | 178tok/1stream/warm | 0.665 | 4.949 | 5.157 |
| explicit 0.40 | 178tok/2stream/fresh | 0.512 | 0.231 | 0.732 |
| explicit 0.40 | 178tok/2stream/warm | 1.159 | 0.261 | 1.370 |
| explicit 0.40 | 8243tok/1stream/fresh | 0.074 | 0.075 | 0.125 |
| explicit 0.40 | 8243tok/1stream/warm | 0.789 | 1.383 | 1.864 |
| explicit 0.40 | 8243tok/2stream/fresh | 0.084 | 0.034 | 0.117 |
| explicit 0.40 | 8243tok/2stream/warm | 1.253 | 0.881 | 2.006 |
| explicit 0.55 | 178tok/1stream/fresh | 0.951 | 7.482 | 7.981 |
| explicit 0.55 | 178tok/1stream/warm | 1.359 | 5.441 | 6.136 |
| explicit 0.55 | 178tok/2stream/fresh | 0.375 | 0.889 | 1.062 |
| explicit 0.55 | 178tok/2stream/warm | 0.013 | 0.806 | 0.806 |
| explicit 0.55 | 8243tok/1stream/fresh | 0.027 | 0.196 | 0.196 |
| explicit 0.55 | 8243tok/1stream/warm | 0.970 | 1.418 | 1.895 |
| explicit 0.55 | 8243tok/2stream/fresh | 0.130 | 0.118 | 0.197 |
| explicit 0.55 | 8243tok/2stream/warm | 0.667 | 0.685 | 1.175 |

## Full per-arm paired bucket details

Slash separates request-list positions; semicolon separates r0 from r1. Native slot assignment can differ.

| Arm | Bucket | Wall tok/s r0 / r1 | Repeat absolute difference | Client TTFT s r0 ; r1 | API decode tok/s r0 ; r1 |
| --- | --- | --- | --- | --- | --- |
| 00 | 178tok/1stream/fresh | 27.76 / 34.69 | 6.938 | 1.292 ; 1.247 | 43.4 ; 61.6 |
| 00 | 178tok/1stream/warm | 52.42 / 57.88 | 5.453 | 0.189 ; 0.173 | 57.2 ; 63.0 |
| 00 | 178tok/2stream/fresh | 33.98 / 34.90 | 0.922 | 1.278 / 2.670 ; 2.558 / 1.178 | 22.9 / 33.0 ; 33.4 / 23.0 |
| 00 | 178tok/2stream/warm | 44.95 / 45.55 | 0.603 | 0.159 / 0.523 ; 0.151 / 0.474 | 24.3 / 26.2 ; 36.4 / 39.5 |
| 00 | 8243tok/1stream/fresh | 11.30 / 11.36 | 0.058 | 6.961 ; 6.938 | 61.2 ; 62.1 |
| 00 | 8243tok/1stream/warm | 55.39 / 55.81 | 0.425 | 0.205 ; 0.195 | 61.4 ; 61.5 |
| 00 | 8243tok/2stream/fresh | 10.41 / 10.40 | 0.006 | 16.098 / 7.014 ; 16.022 / 6.989 | 60.7 / 9.3 ; 59.2 / 9.2 |
| 00 | 8243tok/2stream/warm | 42.63 / 43.56 | 0.930 | 0.598 / 0.228 ; 0.563 / 0.202 | 25.3 / 23.4 ; 25.5 / 23.7 |
| 01 | 178tok/1stream/fresh | 26.97 / 35.57 | 8.599 | 1.296 ; 1.221 | 41.5 ; 63.5 |
| 01 | 178tok/1stream/warm | 51.79 / 57.20 | 5.411 | 0.185 ; 0.174 | 56.3 ; 62.3 |
| 01 | 178tok/2stream/fresh | 33.78 / 34.43 | 0.651 | 1.281 / 2.685 ; 1.234 / 2.603 | 22.7 / 32.9 ; 22.8 / 33.0 |
| 01 | 178tok/2stream/warm | 44.20 / 45.05 | 0.850 | 0.158 / 0.576 ; 0.156 / 0.482 | 24.0 / 26.0 ; 24.1 / 25.9 |
| 01 | 8243tok/1stream/fresh | 11.29 / 11.34 | 0.049 | 6.948 ; 6.950 | 60.4 ; 62.0 |
| 01 | 8243tok/1stream/warm | 54.63 / 55.36 | 0.730 | 0.210 ; 0.197 | 60.6 ; 61.0 |
| 01 | 8243tok/2stream/fresh | 10.33 / 10.38 | 0.046 | 6.996 / 16.119 ; 16.027 / 6.981 | 9.2 / 56.9 ; 59.3 / 9.2 |
| 01 | 8243tok/2stream/warm | 42.25 / 43.30 | 1.051 | 0.216 / 0.595 ; 0.575 / 0.200 | 23.1 / 25.0 ; 25.4 / 23.6 |
| 02 | 178tok/1stream/fresh | 27.98 / 34.87 | 6.896 | 1.286 ; 1.237 | 43.8 ; 61.8 |
| 02 | 178tok/1stream/warm | 53.03 / 57.06 | 4.035 | 0.183 ; 0.169 | 57.7 ; 61.9 |
| 02 | 178tok/2stream/fresh | 33.70 / 33.47 | 0.231 | 2.670 / 1.253 ; 2.652 / 1.244 | 32.5 / 22.5 ; 31.8 / 22.1 |
| 02 | 178tok/2stream/warm | 45.13 / 45.39 | 0.261 | 0.151 / 0.507 ; 0.150 / 0.475 | 36.5 / 39.5 ; 36.4 / 39.4 |
| 02 | 8243tok/1stream/fresh | 11.15 / 11.18 | 0.026 | 7.005 ; 6.988 | 58.5 ; 58.6 |
| 02 | 8243tok/1stream/warm | 54.20 / 55.59 | 1.383 | 0.211 ; 0.187 | 60.1 ; 60.9 |
| 02 | 8243tok/2stream/fresh | 10.37 / 10.41 | 0.032 | 7.045 / 16.140 ; 16.091 / 6.996 | 9.2 / 60.9 ; 59.6 / 9.3 |
| 02 | 8243tok/2stream/warm | 42.73 / 43.35 | 0.625 | 0.562 / 0.201 ; 0.212 / 0.549 | 37.5 / 36.0 ; 36.1 / 38.2 |
| 03 | 178tok/1stream/fresh | 25.70 / 33.19 | 7.482 | 1.321 ; 1.239 | 39.0 ; 56.8 |
| 03 | 178tok/1stream/warm | 46.43 / 51.87 | 5.441 | 0.195 ; 0.186 | 50.2 ; 56.3 |
| 03 | 178tok/2stream/fresh | 31.38 / 31.86 | 0.487 | 1.307 / 2.731 ; 2.619 / 1.215 | 20.7 / 29.0 ; 28.8 / 20.5 |
| 03 | 178tok/2stream/warm | 40.75 / 40.07 | 0.677 | 0.168 / 0.558 ; 0.494 / 0.168 | 21.9 / 23.6 ; 22.7 / 21.4 |
| 03 | 8243tok/1stream/fresh | 11.02 / 10.88 | 0.140 | 6.967 ; 7.018 | 53.9 ; 52.0 |
| 03 | 8243tok/1stream/warm | 48.70 / 48.27 | 0.431 | 0.220 ; 0.218 | 53.6 ; 53.0 |
| 03 | 8243tok/2stream/fresh | 10.17 / 10.19 | 0.017 | 7.031 / 16.164 ; 7.015 / 16.148 | 9.0 / 53.0 ; 9.0 / 53.3 |
| 03 | 8243tok/2stream/warm | 38.79 / 39.48 | 0.685 | 0.231 / 0.618 ; 0.598 / 0.209 | 21.1 / 22.7 ; 33.8 / 33.8 |
| 04 | 178tok/1stream/fresh | 25.21 / 31.78 | 6.576 | 1.338 ; 1.265 | 38.0 ; 53.5 |
| 04 | 178tok/1stream/warm | 45.73 / 49.84 | 4.112 | 0.202 ; 0.202 | 49.6 ; 54.4 |
| 04 | 178tok/2stream/fresh | 30.80 / 31.69 | 0.889 | 2.753 / 1.296 ; 1.255 / 2.683 | 28.3 / 20.2 ; 20.7 / 29.0 |
| 04 | 178tok/2stream/warm | 39.99 / 40.80 | 0.806 | 0.170 / 0.562 ; 0.549 / 0.179 | 33.5 / 34.7 ; 34.8 / 34.1 |
| 04 | 8243tok/1stream/fresh | 10.83 / 11.02 | 0.196 | 7.022 ; 6.975 | 50.9 ; 54.1 |
| 04 | 8243tok/1stream/warm | 46.81 / 48.22 | 1.418 | 0.248 ; 0.225 | 52.1 ; 53.2 |
| 04 | 8243tok/2stream/fresh | 9.99 / 10.11 | 0.118 | 7.142 / 16.344 ; 7.050 / 16.235 | 8.8 / 51.8 ; 8.9 / 52.6 |
| 04 | 8243tok/2stream/warm | 38.63 / 38.30 | 0.332 | 0.232 / 0.649 ; 0.229 / 0.635 | 21.1 / 22.8 ; 20.8 / 22.4 |
| 05 | 178tok/1stream/fresh | 27.25 / 34.08 | 6.834 | 1.318 ; 1.263 | 42.6 ; 60.3 |
| 05 | 178tok/1stream/warm | 51.91 / 56.85 | 4.949 | 0.192 ; 0.179 | 56.6 ; 62.1 |
| 05 | 178tok/2stream/fresh | 32.97 / 33.18 | 0.210 | 2.745 / 1.285 ; 1.295 / 2.689 | 32.0 / 22.0 ; 22.1 / 31.6 |
| 05 | 178tok/2stream/warm | 44.02 / 44.19 | 0.162 | 0.534 / 0.156 ; 0.496 / 0.160 | 25.7 / 23.8 ; 38.0 / 36.0 |
| 05 | 8243tok/1stream/fresh | 11.20 / 11.28 | 0.075 | 6.996 ; 6.983 | 59.6 ; 61.4 |
| 05 | 8243tok/1stream/warm | 53.72 / 54.49 | 0.767 | 0.217 ; 0.203 | 59.7 ; 60.2 |
| 05 | 8243tok/2stream/fresh | 10.29 / 10.32 | 0.034 | 7.048 / 16.120 ; 16.185 / 7.038 | 9.2 / 55.7 ; 58.8 / 9.2 |
| 05 | 8243tok/2stream/warm | 41.35 / 42.23 | 0.881 | 0.651 / 0.212 ; 0.604 / 0.201 | 37.0 / 34.9 ; 25.0 / 23.1 |
| 06 | 178tok/1stream/fresh | 25.94 / 35.54 | 9.596 | 1.326 ; 1.250 | 39.6 ; 64.6 |
| 06 | 178tok/1stream/warm | 50.47 / 56.47 | 6.001 | 0.199 ; 0.186 | 55.0 ; 61.8 |
| 06 | 178tok/2stream/fresh | 32.86 / 33.29 | 0.423 | 1.333 / 2.776 ; 1.298 / 2.699 | 22.2 / 32.1 ; 22.2 / 32.0 |
| 06 | 178tok/2stream/warm | 44.26 / 44.69 | 0.434 | 0.540 / 0.164 ; 0.156 / 0.500 | 39.0 / 36.1 ; 24.0 / 25.8 |
| 06 | 8243tok/1stream/fresh | 11.02 / 11.17 | 0.154 | 7.048 ; 7.012 | 56.2 ; 59.3 |
| 06 | 8243tok/1stream/warm | 52.94 / 54.32 | 1.378 | 0.218 ; 0.202 | 58.8 ; 59.9 |
| 06 | 8243tok/2stream/fresh | 10.32 / 10.33 | 0.011 | 7.060 / 16.162 ; 16.123 / 7.029 | 9.1 / 60.1 ; 57.5 / 9.2 |
| 06 | 8243tok/2stream/warm | 43.00 / 42.90 | 0.096 | 0.565 / 0.211 ; 0.208 / 0.575 | 38.0 / 36.1 ; 35.7 / 38.0 |
| 07 | 178tok/1stream/fresh | 27.52 / 34.60 | 7.083 | 1.305 ; 1.257 | 43.0 ; 61.8 |
| 07 | 178tok/1stream/warm | 51.36 / 57.43 | 6.074 | 0.199 ; 0.185 | 56.2 ; 63.0 |
| 07 | 178tok/2stream/fresh | 33.99 / 34.39 | 0.396 | 1.283 / 2.686 ; 1.237 / 2.620 | 22.9 / 33.3 ; 22.8 / 33.0 |
| 07 | 178tok/2stream/warm | 45.74 / 45.35 | 0.394 | 0.514 / 0.155 ; 0.487 / 0.157 | 40.2 / 37.3 ; 39.4 / 36.5 |
| 07 | 8243tok/1stream/fresh | 11.18 / 11.32 | 0.137 | 6.990 ; 6.965 | 58.8 ; 61.9 |
| 07 | 8243tok/1stream/warm | 54.50 / 56.07 | 1.569 | 0.214 ; 0.199 | 60.6 ; 62.0 |
| 07 | 8243tok/2stream/fresh | 10.38 / 10.43 | 0.051 | 7.034 / 16.158 ; 7.021 / 16.086 | 9.2 / 61.3 ; 9.2 / 61.9 |
| 07 | 8243tok/2stream/warm | 42.41 / 42.84 | 0.423 | 0.204 / 0.547 ; 0.198 / 0.536 | 23.0 / 24.7 ; 23.2 / 24.9 |

## Limits

- Fixed reload order, no independent window replication/randomization.
- r0 solo-before-dual, r1 dual-before-solo: cache/order and MTP paths differ.
- Probed baseline first and explicit 0.25 last are different startup modes/order positions; not pooled.
- API/native timing and hit/PCIe counters may be segment scoped; hit denominator changes with configured split.
- Latest native/API 12-row snapshot covers repeat 1 only, not all 24 requests.
- Integer fixture is not general workload, token/logit parity, full-context/vision/pressure/handoff gate.
- No cache flush, clocks/affinity/worker/reserve change. Startup probe is not sustained inference transfer bandwidth.
- Sampled peaks/minima are not sub-second extrema; audio latency not measured.
