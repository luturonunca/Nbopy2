#!/bin/bash

# Exit if no commit message is provided
if [ -z "$1" ]; then
  echo "Usage: ./commitall.sh \"Your commit message here\""
  exit 1
fi

# Add all changes
git add .

# Commit with provided message
git commit -m "$1"

# Get current branch name
branch=$(git symbolic-ref --short HEAD)

# Push to current branch
git push -u origin "$branch"
