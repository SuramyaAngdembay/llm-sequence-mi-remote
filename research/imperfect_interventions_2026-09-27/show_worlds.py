import json, sys
W = json.load(open(sys.argv[1]))
print(f"{'condition':<28}{'full':>7}{'tau':>7}{'I2':>8}{'I1':>8}{'applied':>8}{'necess':>8}{'inter':>7}{'altSw':>7}{'feas':>5}{'realE':>7}{'d_hat':>7}{'Kc':>6}{'Ks':>6}{'ineq':>5}{'rej':>5}")
for c, ws in W.items():
    for w in ws:
        g = lambda k: w.get(k, float('nan'))
        print(f"{c:<28}{g('full'):7.3f}{g('tau'):7.3f}{g('ideal2'):8.4f}{g('ideal1'):8.4f}{g('applied'):8.4f}{g('necessity'):8.4f}{g('interaction'):7.3f}{g('alt_swapin'):7.3f}{g('share_feasible'):5.2f}{g('mean_realization_error'):7.3f}{g('mean_d_hat_to_ideal2'):7.3f}{g('mean_Kc'):6.2f}{g('mean_Ks'):6.2f}{g('share_ineq_active'):5.2f}{g('rejected_infeasible_units'):5d}")
