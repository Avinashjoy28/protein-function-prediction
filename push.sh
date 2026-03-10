#!/bin/bash
echo "=== Pushing updates ==="

git add .

if git diff --cached --name-only | grep -q "api.json\|\.env"; then
    echo "ERROR: Secrets file staged! Aborting."
    git reset
    exit 1
fi

echo "Enter commit message:"
read MSG
git commit -m "$MSG"
git push

echo "=== Pushed! Visit https://github.com/Avinashjoy28/protein-function-prediction ==="
