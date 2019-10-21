#!/usr/bin/env bash


libdir=$(python3 -m site --user-site)
echo $libdir

ln -s -v -f $HOME/Dropbox/linux/script/mplib/qq/qq $libdir
