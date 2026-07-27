/**
 * Sliding-window completion-rate meter used by browser-driven Runtime loops.
 *
 * The old UI measured one full cycle of "infer_once + snapshot JPEG + image
 * decode + canvas render".  That made the displayed inference FPS and the next
 * inference request depend on Web image refresh latency.  This meter records
 * only completed inference requests, so it represents the rate at which the
 * Runtime actually receives and completes inference work.
 */
export class SlidingRateMeter {
  constructor(windowMs = 2000, maxSamples = 180) {
    this.windowMs = Math.max(250, Number(windowMs) || 2000);
    this.maxSamples = Math.max(2, Math.round(Number(maxSamples) || 180));
    this.samples = [];
  }

  reset() {
    this.samples.length = 0;
  }

  mark(timestamp = performance.now()) {
    const now = Number(timestamp);
    if (!Number.isFinite(now)) return this.value();
    this.samples.push(now);
    this.#prune(now);
    return this.value();
  }

  value() {
    if (this.samples.length < 2) return 0;
    const first = this.samples[0];
    const last = this.samples[this.samples.length - 1];
    const elapsedMs = last - first;
    return elapsedMs > 0 ? ((this.samples.length - 1) * 1000) / elapsedMs : 0;
  }

  #prune(now) {
    const cutoff = now - this.windowMs;
    while (this.samples.length > 2 && this.samples[0] < cutoff) this.samples.shift();
    if (this.samples.length > this.maxSamples) {
      this.samples.splice(0, this.samples.length - this.maxSamples);
    }
  }
}

export function targetIntervalMs(fps, fallbackMs = 200) {
  const number = Number(fps);
  if (!Number.isFinite(number) || number <= 0) return Math.max(16, Number(fallbackMs) || 200);
  return Math.max(16, Math.round(1000 / number));
}
