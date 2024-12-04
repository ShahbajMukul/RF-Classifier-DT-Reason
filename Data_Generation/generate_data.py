import pandas as pd
import numpy as np
import random

# Seed for reproducibility
np.random.seed(42)

# Machine Types and Probabilities
machine_types = ["Sidel Blowmolder", "Injection Molder", "Labeler", "Filler"]
machine_probs = [0.3, 0.25, 0.25, 0.2]

# Codes and Probabilities per Machine Type
codes_per_machine = {
    "Sidel Blowmolder": (["Color Change", "Material Jam", "General Maintenance Required"], [0.5, 0.3, 0.2]),
    "Injection Molder": (["Loose Bolt", "Material Jam", "General Maintenance Required"], [0.6, 0.3, 0.1]),
    "Labeler": (["Loose Bolt", "General Maintenance Required"], [0.4, 0.6]),
    "Filler": (["Material Jam", "Color Change", "General Maintenance Required"], [0.7, 0.2, 0.1])
}

# Duration Ranges per Code
duration_ranges = {
    "Color Change": (5, 15),
    "Loose Bolt": (10, 30),
    "Material Jam": (20, 60),
    "General Maintenance Required": (60, 120)
}

# Machine Locations per Machine Type
locations_per_machine = {
    "Sidel Blowmolder": ["Production Line 1", "Production Line 2"],
    "Injection Molder": ["Production Line 1"],
    "Labeler": ["Production Line 2"],
    "Filler": ["Downtime Area", "Production Line 1"]
}

# Groups per Machine Type
groups_per_machine = {
    "Sidel Blowmolder": "Blowmolder Section",
    "Injection Molder": "Injection Section",
    "Labeler": "Labeling Unit",
    "Filler": "Filling Station"
}


data = []

for _ in range(2000):
    # Machine Type
    machine = np.random.choice(machine_types, p=machine_probs)
    
    # Error Code (Z4 event)
    error_code = 100
    
    # Time of Day
    time_of_day = np.random.randint(0, 24)
    minute = np.random.randint(0, 60)
    start_time = time_of_day * 60 + minute  # Convert to minutes for simplicity
    
    # Machine Location
    location = random.choice(locations_per_machine[machine])
    
    # Group
    group = groups_per_machine[machine]
    
    # Code
    codes, code_probs = codes_per_machine[machine]
    code = np.random.choice(codes, p=code_probs)
    
    # Duration
    duration_min, duration_max = duration_ranges[code]
    duration = np.random.randint(duration_min, duration_max + 1)
    
    # End Time
    end_time = start_time + duration  # Assuming no cross-day downtimes
    
    # Compile Record
    record = {
        "MachineType": machine,
        "ErrorCode": error_code,
        "TimeOfDay": time_of_day,
        "StartTime": start_time,
        "EndTime": end_time,
        "Duration": duration,
        "MachineLocation": location,
        "Group": group,
        "Code": code
    }
    
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)


print("Machine Type Distribution:")
print(df['MachineType'].value_counts(normalize=True))

print("\nCode Distribution:")
print(df['Code'].value_counts(normalize=True))

print("\nGroup Distribution:")
print(df['Group'].value_counts(normalize=True))


for code in duration_ranges.keys():
    durations = df[df['Code'] == code]['Duration']
    min_dur, max_dur = duration_ranges[code]
    if not durations.between(min_dur, max_dur).all():
        print(f"Duration out of range for code: {code}")