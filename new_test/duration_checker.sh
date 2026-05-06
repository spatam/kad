#!/usr/bin/bash

soxi -D *.wav | awk '{ sum += $3; n++ } END { if (n > 0) print "Avg duration: " sum / n " sec"; print "Total files: " n }'
