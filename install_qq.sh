#!/usr/bin/env bash

scname=$(basename -- "$(readlink -f -- "$0")")
scdir="$(dirname $(readlink -f $0))"

bash $scdir/update_requirements.sh
bash $scdir/make_site_packages_symlink.sh
