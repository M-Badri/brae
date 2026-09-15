    Comparison refreshCmp;
    auto checkRefreshBuffer = [&](const DeviceBuffer<scalar>& production,
                                  const DeviceBuffer<scalar>& factory)
    {
        std::vector<scalar> p, f;
        production.copyTo(p);
        factory.copyTo(f);
        const Comparison c = compare(p, f);
        refreshCmp.bitwise = refreshCmp.bitwise && c.bitwise;
        refreshCmp.differing += c.differing;
        refreshCmp.maxAbs = std::max(refreshCmp.maxAbs, c.maxAbs);
        refreshCmp.maxRel = std::max(refreshCmp.maxRel, c.maxRel);
    };
    checkRefreshBuffer(lc.fwd, factoryLc.fwd);
    checkRefreshBuffer(lc.bwd, factoryLc.bwd);
    checkRefreshBuffer(cc.fwdDiag, factoryCc.fwdDiag);
    checkRefreshBuffer(cc.bwdDiag, factoryCc.bwdDiag);
    checkRefreshBuffer(cc.fwdB, factoryCc.fwdB);
    checkRefreshBuffer(cc.bwdB, factoryCc.bwdB);
    std::printf("  scalar refresh identity: %s; differing %zu, max abs %.3e, max rel %.3e\n",
                refreshCmp.bitwise ? "BITWISE" : "DIFF", refreshCmp.differing,
                static_cast<double>(refreshCmp.maxAbs), static_cast<double>(refreshCmp.maxRel));

    const Timing refreshTime = medianTiming(
        [] {}, [&] { gsLevelCoefsRefresh(A, lv, lc); gsLevelCellsRefresh(A, dB, lv, cc); },
        [] {}, [&] { factoryGsLevelCoefsRefresh(A, lv, factoryLc); factoryGsLevelCellsRefresh(A, dB, lv, factoryCc); },
        timingSweeps);
    std::printf("  scalar refresh median: production %.3f ms, factory %.3f ms, speedup %.3fx\n",
                refreshTime.productionMs, refreshTime.factoryMs,
                refreshTime.productionMs/refreshTime.factoryMs);
