#!/bin/bash
# Profile Chisel elaboration and synthesis

echo "=== Profiling Chisel Elaboration ==="
echo "Starting at: $(date)"

# Time the elaboration
/usr/bin/time -v sbt "runMain empty.Main" 2>&1 | tee elaboration_profile.txt

echo ""
echo "=== Generated File Analysis ==="
ls -lh generated/

echo ""
echo "=== Line Count Breakdown ==="
wc -l generated/*.sv generated/*.fir 2>/dev/null

echo ""
echo "=== RTL Statistics ==="
echo "Multipliers (*):"
grep -o '\*' generated/Pipeline.sv | wc -l

echo "Always blocks:"
grep -c 'always @' generated/Pipeline.sv

echo "Assign statements:"
grep -c '^  assign' generated/Pipeline.sv

echo "Wire declarations:"
grep -c '^  wire' generated/Pipeline.sv

echo "Reg declarations:"
grep -c '^  reg' generated/Pipeline.sv

echo ""
echo "=== Module Hierarchy (by size) ==="
awk '/^module/{mod=$2; start=NR} /^endmodule/{print NR-start, mod}' generated/Pipeline.sv | sort -rn | head -20

echo ""
echo "=== Module Count ==="
grep '^module' generated/Pipeline.sv | awk '{print $2}' | sort | uniq -c | sort -rn

echo ""
echo "=== Width Analysis (bit counts in signals) ==="
grep -oP '\[\d+:\d+\]' generated/Pipeline.sv | head -20

echo ""
echo "Peak memory usage and elapsed time in elaboration_profile.txt"
grep -E "Maximum resident|User time|System time|Elapsed" elaboration_profile.txt
