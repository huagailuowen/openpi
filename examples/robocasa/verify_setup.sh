#!/bin/bash

# Quick verification script for RoboCasa OpenPI integration

echo "🔍 RoboCasa OpenPI Integration Verification"
echo "===========================================" 
echo ""

# Check if all necessary files exist
echo "📁 Checking integration files..."

FILES=(
    "env.py"
    "main.py"
    "saver.py"
    "requirements.txt"
    "README.md"
    "test_integration.py"
    "setup_integration.sh"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
    fi
done

echo ""
echo "🐍 Checking Python syntax..."

# Check Python syntax for main files
for pyfile in env.py main.py saver.py test_integration.py; do
    if [ -f "$pyfile" ]; then
        if python3 -m py_compile "$pyfile" 2>/dev/null; then
            echo "✅ $pyfile (syntax OK)"
        else
            echo "❌ $pyfile (syntax error)"
        fi
    fi
done

echo ""
echo "📋 Summary:"
echo "This integration provides:"
echo "• RoboCasa environment adapter for OpenPI"
echo "• WebSocket client for policy communication"  
echo "• Video recording capabilities"
echo "• Support for all RoboCasa kitchen tasks"
echo "• Compatible with OpenPI action chunking"
echo ""

echo "🚀 Ready to test!"
echo "Run: ./test_integration.py"
echo ""
