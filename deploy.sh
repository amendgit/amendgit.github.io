
#!/bin/sh
# https://gohugo.io/hosting-and-deployment/hosting-on-github/

DIR=$(dirname "$0")

echo "Deleting old publication"
rm -rf public
mkdir public
git worktree prune
rm -rf .git/worktrees/public/

echo "Checking out master branch into public"
git worktree add -B master public origin/master

echo "Removing existing files"
rm -rf public/*

echo "Generating site"
hugo

echo "Updating master branch"
cd public && git add --all && git commit -m "Publishing to master (deploy.sh)" && git push origin master