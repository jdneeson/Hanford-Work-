#!/bin/bash
set -euo pipefail

cache_path="${_CONDOR_SCRATCH_DIR}/krb5cc"
export KRB5CCNAME="FILE:${cache_path}"

cleanup() {
    rm -f "$PWD/ligo.org.keytab" "$cache_path"
}
trap cleanup EXIT

kinit -kt "$PWD/ligo.org.keytab" \
      james.neeson@LIGO.ORG


exec /cvmfs/software.igwn.org/conda/envs/igwn-py310/bin/python "$@"