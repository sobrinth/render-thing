#!/usr/bin/env bash
# Fetch glTF sample models into assets/models/downloaded (gitignored).
#
# Just fetching models as needed to not have to use a submodule (and the large size)
#
#   scripts/fetch-assets.sh                 # Sponza
#   scripts/fetch-assets.sh Sponza FlightHelmet DamagedHelmet
set -euo pipefail

repo_url="https://github.com/KhronosGroup/glTF-Sample-Assets"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/assets/models/downloaded/glTF-Sample-Assets"

models=("$@")
[[ ${#models[@]} -eq 0 ]] && models=(Sponza)

if [[ ! -d "$dest/.git" ]]; then
    git clone --depth 1 --filter=blob:none --sparse "$repo_url" "$dest"
fi

git -C "$dest" sparse-checkout add "${models[@]/#/Models/}"

for model in "${models[@]}"; do
    echo "$model -> $dest/Models/$model"
done
