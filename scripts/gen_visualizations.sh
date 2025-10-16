#!/bin/bash

# dependencies: yosys, netlistsvg

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERILOG_FILE="${1:-$PROJECT_ROOT/generated/Pipeline.v}"
OUTPUT_DIR="${2:-$PROJECT_ROOT}"

if [ ! -f "$VERILOG_FILE" ]; 
then
    echo "Error: Verilog file not found: $VERILOG_FILE"
    exit 1
fi

echo "Generating visualizations from: $VERILOG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "================================================"

mkdir -p "$OUTPUT_DIR"

# The queues are the FIFOs between the layers
# TODO: We could just read from the generated verilog what the modules are?
modules=("Pipeline" "Queue" "Queue_1" "DenseDataflowFold" "DenseDataflowFold_1")

for module in "${modules[@]}"; do
    echo "Processing module: $module"

    json_file="$OUTPUT_DIR/${module,,}.json"
    svg_file="$OUTPUT_DIR/${module,,}.svg"

    # Generate JSON with Yosys
    yosys -p "read_verilog $VERILOG_FILE; hierarchy -top $module; proc; write_json $json_file" 2>&1 | grep -E "(Error|Warning|Top module)"

    if [ -f "$json_file" ]; then
        echo "  Generated: $json_file"

        # Convert to SVG with netlistsvg
        netlistsvg "$json_file" -o "$svg_file" 2>&1 | grep -v "^$"

        if [ -f "$svg_file" ]; then
            size=$(du -h "$svg_file" | cut -f1)
            echo "  Generated: $svg_file ($size)"
        else
            echo "  Failed to generate SVG"
        fi
    else
        echo "  Failed to generate JSON"
    fi
done

echo "================================================"
echo "Visualization complete!"
echo
echo "Generated files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/*.svg 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
