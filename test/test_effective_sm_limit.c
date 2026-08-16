/*
 * GPU-free unit test for effective SM limit resolution (elastic SM step-3).
 *
 * Verifies get_effective_sm_limit_on():
 *   - dynamic_sm_limit wins when non-zero
 *   - dynamic==0 falls back to floor_sm_limit
 *   - floor==0 falls back to sm_limit
 *   - all-zero means unlimited (0)
 *   - NULL region / bad device index return 0
 *   - re-read after mutating dynamic sees the new value (watcher tick wiring)
 */
#include <stdio.h>
#include <string.h>

#include "multiprocess/multiprocess_memory_limit.h"

static int failures;

static void expect_eq(int got, int want, const char *msg) {
    if (got != want) {
        fprintf(stderr, "FAIL: %s (got=%d want=%d)\n", msg, got, want);
        failures++;
    }
}

int main(void) {
    shared_region_t region;
    const int dev = 0;

    failures = 0;
    memset(&region, 0, sizeof(region));

    expect_eq(get_effective_sm_limit_on(NULL, dev), 0, "NULL region -> 0");
    expect_eq(get_effective_sm_limit_on(&region, -1), 0, "negative device -> 0");
    expect_eq(get_effective_sm_limit_on(&region, CUDA_DEVICE_MAX_COUNT), 0,
              "device out of range -> 0");

    region.sm_limit[dev] = 50;
    region.floor_sm_limit[dev] = 40;
    region.dynamic_sm_limit[dev] = 70;
    expect_eq(get_effective_sm_limit_on(&region, dev), 70,
              "non-zero dynamic wins over floor/sm");

    region.dynamic_sm_limit[dev] = 0;
    expect_eq(get_effective_sm_limit_on(&region, dev), 40,
              "dynamic==0 falls back to floor");

    region.floor_sm_limit[dev] = 0;
    expect_eq(get_effective_sm_limit_on(&region, dev), 50,
              "floor==0 falls back to sm_limit");

    region.sm_limit[dev] = 0;
    expect_eq(get_effective_sm_limit_on(&region, dev), 0,
              "all zero means unlimited/no limit");

    /* Simulate watcher re-read across ticks: mutate dynamic and read again. */
    region.sm_limit[dev] = 30;
    region.floor_sm_limit[dev] = 30;
    region.dynamic_sm_limit[dev] = 30;
    expect_eq(get_effective_sm_limit_on(&region, dev), 30, "tick1 baseline");

    region.dynamic_sm_limit[dev] = 80;
    expect_eq(get_effective_sm_limit_on(&region, dev), 80,
              "tick2 must observe raised dynamic");

    region.dynamic_sm_limit[dev] = 0;
    expect_eq(get_effective_sm_limit_on(&region, dev), 30,
              "tick3 dynamic cleared falls back to floor");

    /* Independent per-device: writing device 1 must not affect device 0. */
    region.dynamic_sm_limit[1] = 90;
    region.floor_sm_limit[1] = 20;
    region.sm_limit[1] = 20;
    expect_eq(get_effective_sm_limit_on(&region, 0), 30,
              "device 0 unchanged when device 1 updates");
    expect_eq(get_effective_sm_limit_on(&region, 1), 90,
              "device 1 reads its own dynamic");

    if (failures != 0) {
        fprintf(stderr, "%d assertion(s) failed\n", failures);
        return 1;
    }
    printf("test_effective_sm_limit: PASS\n");
    return 0;
}
