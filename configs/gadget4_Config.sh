#!/usr/bin/env bash
# GADGET-4 Config.sh — feature flags for the build.
# Copy this to gadget4/Config.sh before running make.

#-- Output format
OUTPUT_FORMAT_HDF5=1

#-- Gravity
SELFGRAVITY=1
PMGRID=256          # enable PM grid (TreePM); increase for larger boxes

#-- Units: 1 kpc, 1e10 Msun, internal velocity units
GADGET_LONG_LONG_PARTICLEID=1   # particle IDs as 64-bit

#-- Parallelism
OPENMP=1

#-- Initial conditions in HDF5
HAVE_HDF5=1

#-- Cosmological sim OFF by default (isolated galaxy mode)
# PERIODIC — enable for cosmological runs
