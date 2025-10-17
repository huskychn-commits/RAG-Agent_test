import math

P0 = 1.0e5      # Pa
Vc = 0.01       # m³ (10 L)
VpM = 0.001     # m³ (1 L)
N = int(input("Enter number of strokes: "))

P = P0
total_work = 0.0

for k in range(N):
    PN = P0 * (1 + 0.1 * k)

    # Step 1: find v* where compression triggers valve opening
    v_star = VpM * P0 / PN
    if v_star > VpM:
        v_star = VpM
    elif v_star < 0:
        v_star = 0

    # Work done in sealed compression
    if PN > P0:
        W1 = P0 * VpM * math.log(PN / P0)
    else:
        W1 = 0.0  # ln(≤1) but only if v_star < VpM

    # After first balance
    numerator = PN * Vc + P0 * VpM
    denominator = Vc + v_star
    P_prime = numerator / denominator

    # Work done pushing remaining volume at pressure P'
    W2 = P_prime * v_star

    Wk = W1 + W2
    total_work += Wk

    print(f"Stroke {k+1}: W = {Wk:.2f} J, total = {total_work:.2f} J")

print(f"Final pressure: {PN + 0.1*P0:.2e} Pa")