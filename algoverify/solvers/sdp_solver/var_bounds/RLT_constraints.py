
def RLT_constraints(xyT, x, lx, ux, y, ly, uy):
    return [
        # xyT - x @ ly.T - lx @ y.T + lx @ ly.T >= 0,
        # x @ uy.T - xyT - lx @ uy.T + lx @ y.T >= 0,
        ux @ y.T - ux @ ly.T - xyT + x @ ly.T >= 0,
        ux @ uy.T - ux @ y.T - x @ uy.T + xyT >= 0,
    ]
