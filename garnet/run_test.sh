#!/usr/bin/env bash
# Re-run stage 5b on every sheet fixture, into a freshly cleared output/.
set -u
cd /Users/maetee/Code/GARNET || exit 1

SHEETS=(
  14780-8120-25-25-0002A
  14780-8120-25-25-0003
  14780-8120-25-25-0004
  14780-8120-25-25-0004A
  14780-8120-25-25-0007
  14780-8120-25-25-0007A
  14780-8120-25-25-0008
  14780-8120-25-25-0009
  14780-8120-25-25-0010
)

echo "=== clearing output/ ==="
rm -rf output/*/
echo "remaining entries: $(ls -A output/ 2>/dev/null | wc -l | tr -d ' ')"

echo
echo "=== running ${#SHEETS[@]} sheets ==="
for S in "${SHEETS[@]}"; do
  if .venv/bin/python -m garnet.stage5b.run \
       --stem "$S" --indir garnet/tests/input --out "output/$S" >/dev/null 2>&1; then
    printf "%-24s ok\n" "${S: -4}"
  else
    printf "%-24s FAILED\n" "${S: -4}"
  fi
done
echo "ALL DONE"