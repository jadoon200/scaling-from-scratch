"""Chip peak specs for roofline analysis (Apple M3 Pro, 18-core GPU)."""

PEAK_FLOPS_FP16 = 6.4e12   # FLOP/s
PEAK_BW_BYTES = 150e9      # bytes/s
RIDGE_POINT = PEAK_FLOPS_FP16 / PEAK_BW_BYTES  # ~43 FLOP/byte

if __name__ == "__main__":
    print(f"M3 Pro: {PEAK_FLOPS_FP16/1e12:.1f} TFLOP/s, "
          f"{PEAK_BW_BYTES/1e9:.0f} GB/s, ridge {RIDGE_POINT:.0f} FLOP/byte")
