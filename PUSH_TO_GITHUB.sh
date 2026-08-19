#!/bin/bash
# ============================================================
# Push Delhivery Network Intelligence to GitHub
# ============================================================
# Run this from inside the delhivery-network-intelligence/ folder
# after cloning or extracting the ZIP.
#
# Prerequisites:
#   git config --global user.name  "Your Name"
#   git config --global user.email "your@email.com"
#
# Usage:
#   chmod +x PUSH_TO_GITHUB.sh
#   ./PUSH_TO_GITHUB.sh https://github.com/YOUR_USERNAME/delhivery-network-intelligence.git
# ============================================================

set -e

REMOTE_URL="${1:-https://github.com/YOUR_USERNAME/delhivery-network-intelligence.git}"

echo "🔗 Adding remote: $REMOTE_URL"
git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"

echo "📤 Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo "   Repository: $REMOTE_URL"
echo "   Branch: main"
echo "   Commits: $(git rev-list --count HEAD)"
echo ""
echo "Next steps:"
echo "  1. Visit your repository on GitHub"
echo "  2. Add a description: 'Graph-powered delivery ETA optimization for logistics networks'"
echo "  3. Add topics: graph-ml, logistics, eta-prediction, scikit-learn, networkx, streamlit"
echo "  4. Pin the repository to your profile"
