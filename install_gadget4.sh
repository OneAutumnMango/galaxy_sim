#!/usr/bin/env bash
# install_gadget4.sh — install all deps and build GADGET-4 for Ubuntu/Debian.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GADGET_DIR="$SCRIPT_DIR/gadget4"
BUILD_THREADS=${BUILD_THREADS:-$(nproc)}

# ── 1. System dependencies ─────────────────────────────────────────────────
echo "=== Installing system dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y \
    git build-essential \
    libopenmpi-dev openmpi-bin \
    libhdf5-dev libhdf5-mpi-dev \
    libgsl-dev \
    libfftw3-dev libfftw3-mpi-dev \
    pkg-config

# ── 2. Detect library paths via pkg-config ─────────────────────────────────
echo "=== Detecting library paths ==="

GSL_INCL="$(pkg-config --cflags-only-I gsl 2>/dev/null || echo '-I/usr/include')"
GSL_LIBS="$(pkg-config --libs-only-L gsl 2>/dev/null || echo '-L/usr/lib/x86_64-linux-gnu')"

FFTW_INCL="$(pkg-config --cflags-only-I fftw3 2>/dev/null || echo '-I/usr/include')"
FFTW_LIBS="$(pkg-config --libs-only-L fftw3 2>/dev/null || echo '-L/usr/lib/x86_64-linux-gnu')"

# HDF5 on Ubuntu lives under hdf5/serial or hdf5/openmpi
HDF5_PC=""
for pc in hdf5-serial hdf5 hdf5-openmpi; do
    if pkg-config --exists "$pc" 2>/dev/null; then
        HDF5_PC="$pc"; break
    fi
done
if [[ -n "$HDF5_PC" ]]; then
    HDF5_INCL="$(pkg-config --cflags-only-I "$HDF5_PC")"
    HDF5_LIBS="$(pkg-config --libs-only-L "$HDF5_PC")"
else
    # fallback: probe common locations
    for d in /usr/include/hdf5/serial /usr/include/hdf5 /usr/include; do
        if [[ -f "$d/hdf5.h" ]]; then HDF5_INCL="-I$d"; break; fi
    done
    for d in /usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu/hdf5 /usr/lib/x86_64-linux-gnu; do
        if ls "$d"/libhdf5*.so &>/dev/null 2>&1; then HDF5_LIBS="-L$d"; break; fi
    done
fi

echo "  GSL_INCL  : $GSL_INCL"
echo "  GSL_LIBS  : $GSL_LIBS"
echo "  FFTW_INCL : $FFTW_INCL"
echo "  FFTW_LIBS : $FFTW_LIBS"
echo "  HDF5_INCL : $HDF5_INCL"
echo "  HDF5_LIBS : $HDF5_LIBS"

# ── 3. Clone if not present ────────────────────────────────────────────────
if [[ ! -d "$GADGET_DIR" ]]; then
    echo "=== Cloning GADGET-4 ==="
    git clone --depth=1 https://gitlab.mpcdf.mpg.de/vrs/gadget4.git "$GADGET_DIR"
fi

# ── 4. Write a custom path Makefile for this machine ──────────────────────
echo "=== Writing build configuration ==="
cat > "$GADGET_DIR/buildsystem/Makefile.path.ubuntu" <<EOF
GSL_INCL   = $GSL_INCL
GSL_LIBS   = $GSL_LIBS
FFTW_INCL  = $FFTW_INCL
FFTW_LIBS  = $FFTW_LIBS
HDF5_INCL  = $HDF5_INCL
HDF5_LIBS  = $HDF5_LIBS
HWLOC_INCL =
HWLOC_LIBS =
EOF

# Add ubuntu SYSTYPE to Makefile (idempotent)
if ! grep -q '"ubuntu"' "$GADGET_DIR/Makefile"; then
    sed -i '/ifeq ($(SYSTYPE),"Generic-gcc")/i ifeq ($(SYSTYPE),"ubuntu")\ninclude buildsystem\/Makefile.path.ubuntu\ninclude buildsystem\/Makefile.comp.gcc\nendif\n' \
        "$GADGET_DIR/Makefile"
fi

# Write Makefile.systype
echo 'SYSTYPE="ubuntu"' > "$GADGET_DIR/Makefile.systype"

# ── 5. Write Config.sh ─────────────────────────────────────────────────────
# HDF5 and OpenMP are always enabled in GADGET-4 -- they are NOT Config.sh options.
cat > "$GADGET_DIR/Config.sh" <<'EOFCFG'
SELFGRAVITY
NTYPES=2
DOUBLEPRECISION=1
EVALPOTENTIAL
EOFCFG

# ── 6. Build ───────────────────────────────────────────────────────────────
echo "=== Building GADGET-4 with $BUILD_THREADS threads ==="
cd "$GADGET_DIR"
make -j"$BUILD_THREADS"

echo ""
echo "=== Build complete ==="
echo "Binary: $GADGET_DIR/Gadget4"
