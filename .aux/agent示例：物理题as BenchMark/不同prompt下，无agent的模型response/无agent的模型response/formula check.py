from numpy import log

P0 = 1e5      # Pa
Vc = 0.01     # m³
VpM = 0.001   # m³
r = VpM / Vc  # = 0.1

total_work = 0.0

for k in range(50):  # k = 0 to 49
    x = 1 + 0.1 * k
    term1 = 100 * log(x)
    term2 = 1000 * (1.1 + 0.1*k) * log(1 + 0.1 / x)
    total_work += term1 + term2

print(total_work)