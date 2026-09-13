# Protocol decisions checked on the 2016 validation year

### Check 1: residual (one-step MSE on 2016, standardised units)

```
 variant  persistence  model  ratio  train_s
   plain       0.0818 0.1070 1.3091      125
residual       0.0818 0.0612 0.7486      128
```

### Check 2: five-day roll-out on 2016 (latitude-weighted RMSE)

```
lead_hours                                 6       24       72        120
variable model                                                           
T850     climatology                      3.21    3.21     3.21      3.20
         persistence                      1.19    2.69     3.91      4.19
         roll-out-trained (6000 steps)    1.11    2.30     3.84      4.47
         single-step (2000 steps)         1.33    3.49    13.89     80.07
Z500     climatology                    806.82  806.68   806.23    806.03
         persistence                    219.54  573.14   905.94   1007.51
         roll-out-trained (6000 steps)  185.34  463.37   864.33   1027.18
         single-step (2000 steps)       246.70  730.02  3060.91  16874.17
```

### Check 3: equal-length control (single-step, 6000 steps) on 2016

```
lead_hours                            6       24      72       120
variable model                                                    
T850     single-step (6000 steps)    1.09    2.38    4.15     5.16
Z500     single-step (6000 steps)  181.19  484.41  933.86  1176.99
```
