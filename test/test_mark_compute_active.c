/*
 * GPU-free unit test for launch-path activity marking (elastic SM step-2).
 *
 * Verifies mark_compute_active_on():
 *   - IDLE -> ACTIVE
 *   - last_launch_ns becomes non-zero and is monotonic non-decreasing
 *   - already-ACTIVE stays ACTIVE
 *   - NULL region is a no-op (must not crash)
 */
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

static int failures;

static void expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    }
}

static void sleep_ms(int milliseconds) {
    struct timespec ts;

    ts.tv_sec = milliseconds / 1000;
    ts.tv_nsec = (milliseconds % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

int main(void) {
    shared_region_t region;
    uint64_t t0;
    uint64_t t1;
    uint64_t t2;

    failures = 0;
    memset(&region, 0, sizeof(region));

    atomic_store_explicit(&region.compute_state, COMPUTE_STATE_IDLE,
                          memory_order_relaxed);
    atomic_store_explicit(&region.last_launch_ns, 0, memory_order_relaxed);

    mark_compute_active_on(NULL);
    expect(atomic_load_explicit(&region.compute_state, memory_order_relaxed) ==
               COMPUTE_STATE_IDLE,
           "NULL mark must not mutate a separate region");
    expect(atomic_load_explicit(&region.last_launch_ns, memory_order_relaxed) == 0,
           "NULL mark must leave last_launch_ns at 0");

    mark_compute_active_on(&region);
    expect(atomic_load_explicit(&region.compute_state, memory_order_relaxed) ==
               COMPUTE_STATE_ACTIVE,
           "IDLE must become ACTIVE after mark");
    t0 = atomic_load_explicit(&region.last_launch_ns, memory_order_relaxed);
    expect(t0 != 0, "last_launch_ns must be set to a non-zero timestamp");

    /* Stay ACTIVE when already ACTIVE; timestamp must not go backwards. */
    sleep_ms(2);
    mark_compute_active_on(&region);
    expect(atomic_load_explicit(&region.compute_state, memory_order_relaxed) ==
               COMPUTE_STATE_ACTIVE,
           "ACTIVE must remain ACTIVE");
    t1 = atomic_load_explicit(&region.last_launch_ns, memory_order_relaxed);
    expect(t1 >= t0, "last_launch_ns must be monotonic non-decreasing");

    atomic_store_explicit(&region.compute_state, COMPUTE_STATE_IDLE_CANDIDATE,
                          memory_order_relaxed);
    mark_compute_active_on(&region);
    expect(atomic_load_explicit(&region.compute_state, memory_order_relaxed) ==
               COMPUTE_STATE_ACTIVE,
           "IDLE_CANDIDATE must become ACTIVE after mark");
    t2 = atomic_load_explicit(&region.last_launch_ns, memory_order_relaxed);
    expect(t2 >= t1, "last_launch_ns must not move backwards from candidate");

    if (failures != 0) {
        fprintf(stderr, "%d assertion(s) failed\n", failures);
        return 1;
    }
    printf("test_mark_compute_active: PASS\n");
    return 0;
}
