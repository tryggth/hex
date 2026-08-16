#!/usr/bin/env python3
"""
pack_scripts.py — Pack/Unpack the V5 backend Python toolchain into a single text file.

Usage:
    Pack:    python pack_scripts.py pack  [--output v5_toolchain.txt]
    Unpack:  python pack_scripts.py unpack [--input v5_toolchain.txt]

The packed file uses delimiters of the form:
    ===== FILE: backend/some_script.py =====
to delineate each source file's contents.
"""

import argparse
import os
import sys

DELIMITER_PREFIX = "===== FILE: "
DELIMITER_SUFFIX = " ====="

# V5 toolchain: core architecture, training, data generation, evaluation
V5_FILES = [
    "backend/hex_env.py",
    "backend/muzero_nets.py",
    "backend/latent_mcts.py",
    "backend/classic_mcts.py",
    "backend/train.py",
    "backend/train_supervised.py",
    "backend/generate_expert_data.py",
    "backend/generate_muzero_data.py",
    "backend/transplant.py",
    "backend/arena_logistic.py",
    "backend/arena_sprt.py",
    "backend/verify_v5.py",
]

def pack(output_path, root_dir):
    """Read each source file and write them into a single delimited text file."""
    with open(output_path, "w") as out:
        out.write(f"# V5 Toolchain Pack — {len(V5_FILES)} files\n")
        out.write(f"# Unpack with: python pack_scripts.py unpack --input {os.path.basename(output_path)}\n\n")
        
        for rel_path in V5_FILES:
            abs_path = os.path.join(root_dir, rel_path)
            if not os.path.exists(abs_path):
                print(f"  ⚠️  Skipping (not found): {rel_path}")
                continue
                
            with open(abs_path, "r") as f:
                content = f.read()
                
            out.write(f"{DELIMITER_PREFIX}{rel_path}{DELIMITER_SUFFIX}\n")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            print(f"  ✅ Packed: {rel_path}")
            
    print(f"\n📦 Packed {len(V5_FILES)} files → {output_path}")

def unpack(input_path, root_dir):
    """Read the packed text file and extract each source file back to its original location."""
    with open(input_path, "r") as f:
        content = f.read()
    
    # Split on delimiters
    sections = content.split(DELIMITER_PREFIX)
    files_written = 0
    
    for section in sections[1:]:  # skip the header before the first delimiter
        # The first line after the prefix contains the filename and suffix
        first_newline = section.index("\n")
        header = section[:first_newline]
        
        if not header.endswith(DELIMITER_SUFFIX):
            print(f"  ⚠️  Malformed delimiter, skipping: {header}")
            continue
            
        rel_path = header[: -len(DELIMITER_SUFFIX)]
        file_content = section[first_newline + 1:]
        
        abs_path = os.path.join(root_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        
        with open(abs_path, "w") as f:
            f.write(file_content)
            
        files_written += 1
        print(f"  ✅ Unpacked: {rel_path}")
        
    print(f"\n📂 Unpacked {files_written} files from {input_path}")

def main():
    parser = argparse.ArgumentParser(description="Pack/Unpack V5 backend toolchain")
    subparsers = parser.add_subparsers(dest="command")
    
    pack_parser = subparsers.add_parser("pack", help="Pack source files into a single text file")
    pack_parser.add_argument("--output", type=str, default="v5_toolchain.txt")
    
    unpack_parser = subparsers.add_parser("unpack", help="Unpack a text file back into source files")
    unpack_parser.add_argument("--input", type=str, default="v5_toolchain.txt")
    
    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.command == "pack":
        pack(os.path.join(root_dir, args.output), root_dir)
    elif args.command == "unpack":
        unpack(os.path.join(root_dir, args.input), root_dir)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
