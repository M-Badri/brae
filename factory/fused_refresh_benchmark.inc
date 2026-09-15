    const Timing fusedRefreshTime = medianTiming(
        [] {},
        [&]
        {
            gsLevelCoefsRefresh(A, lv, lc);
            for (int k = 0; k < NC; ++k)
            {
                DeviceLduView Ak = A;
                Ak.diag = diag[k].data();
                gsLevelCellsRefresh(Ak, rhs[k], lv, compCells[k]);
            }
        },
        [] {},
        [&]
        {
            factoryGsLevelCoefsRefresh(A, lv, factoryLc);
            for (int k = 0; k < NC; ++k)
            {
                DeviceLduView Ak = A;
                Ak.diag = diag[k].data();
                factoryGsLevelCellsRefresh(Ak, rhs[k], lv, factoryCompCells[k]);
            }
        },
        timingSweeps);
    std::printf("  fused-3 refresh median: production %.3f ms, factory %.3f ms, speedup %.3fx\n",
                fusedRefreshTime.productionMs, fusedRefreshTime.factoryMs,
                fusedRefreshTime.productionMs/fusedRefreshTime.factoryMs);
