#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERILOG_FILE="${1:-$PROJECT_ROOT/generated/Pipeline.v}"

if [ ! -f "$VERILOG_FILE" ];
then
    echo "Error: Verilog file not found: $VERILOG_FILE"
    echo "Usage: $0 [verilog_file]"
    exit 1
fi

echo "Analyzing design: $VERILOG_FILE"
echo "================================================"
echo

yosys -p "read_verilog $VERILOG_FILE; hierarchy -top Pipeline; stat"
