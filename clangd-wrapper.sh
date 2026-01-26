#!/bin/bash
# Simple wrapper to detect compile_commands.json in subfolders

# Try to find compile_commands.json in one-level subdirectories
for dir in */; do
    if [ -f "$dir/build/compile_commands.json" ]; then
        exec clangd --compile-commands-dir="$dir/build" "$@"
        exit
    fi
done

# fallback
exec clangd "$@"
