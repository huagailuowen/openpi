#!/bin/bash
# Script to generate charts from RoboCasa results

# Make sure the visualization script is executable
chmod +x visualize_results.py

# Install required packages if needed
echo "Installing required packages..."
pip install matplotlib pandas seaborn

# Create sample results for demonstration (if no real results exist)
if [ ! -f "data/robocasa/results.json" ]; then
    echo "No results.json found, creating sample data..."
    mkdir -p data/robocasa
    python3 -c "
import json
from datetime import datetime, timedelta
import random

# Create sample results based on the task list you provided
tasks = [
    'Close Double Door', 'Close Drawer', 'Close Single Door', 'Coffee Press Button',
    'Coffee Serve Mug', 'Coffee Setup Mug', 'Open Double Door', 'Open Drawer',
    'Open Single Door', 'PnP from Cab to Counter', 'PnP from Counter to Cab',
    'PnP from Counter to Microwave', 'PnP from Counter to Sink', 'PnP from Counter to Stove',
    'PnP from Microwave to Counter', 'PnP from Sink to Counter', 'PnP from Stove to Counter',
    'Turn Off Microwave', 'Turn Off Sink Faucet', 'Turn Off Stove', 'Turn On Microwave',
    'Turn On Sink Faucet', 'Turn On Stove', 'Turn Sink Spout'
]

# Sample success rates from your data (converted to percentage)
sample_rates = [18, 84, 48, 38, 8, 0, 8, 12, 24, 4, 6, 0, 2, 0, 0, 2, 4, 48, 62, 6, 30, 26, 10, 28]

results = []
base_time = datetime.now() - timedelta(days=7)

for i, (task, rate) in enumerate(zip(tasks[:12], sample_rates[:12])):  # Use first 12 tasks
    # Convert rate from percentage to decimal
    success_rate = rate / 100.0
    # Add some randomness
    test_times = random.randint(8, 12)
    success_times = int(success_rate * test_times)
    actual_rate = success_times / test_times if test_times > 0 else 0
    
    # Create multiple runs for some tasks
    for run in range(random.randint(1, 3)):
        timestamp = base_time + timedelta(hours=i*2 + run*0.5)
        results.append({
            'timestamp': timestamp.isoformat(),
            'task_name': task,
            'success_times': success_times + random.randint(-1, 1),
            'test_times': test_times,
            'success_rate': max(0, min(1, actual_rate + random.uniform(-0.1, 0.1))),
            'result_string': f'{success_times}/{test_times}'
        })

with open('data/robocasa/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Created sample results with {len(results)} entries')
"
fi

# Generate all types of charts
echo "Generating visualizations..."
python3 visualize_results.py --results-file data/robocasa/results.json --output-dir data/robocasa/charts --chart-types all

echo "Charts generated in data/robocasa/charts/"
echo "Available charts:"
echo "  - success_rate_by_task.png: Bar chart of average success rates"
echo "  - success_rate_timeline.png: Timeline showing performance over time"
echo "  - detailed_task_comparison.png: Individual run comparisons per task"
echo "  - success_rate_heatmap.png: Heatmap of performance patterns"
echo "  - summary_statistics.png: Statistical summary table"
echo "  - summary_statistics.csv: Exportable statistics"
