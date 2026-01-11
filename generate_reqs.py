import subprocess
import re
import sys

# Get pip freeze output
raw_reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')

clean_reqs = []
for line in raw_reqs.splitlines():
    # Remove blank lines
    if not line.strip():
        continue
        
    # Check for @ file:// or similar path dependencies
    # Pattern: package_name @ file:///...
    match = re.match(r'^([^=@\s]+)\s*@\s*.*$', line)
    
    if match:
        package_name = match.group(1)
        # We need to find the version for this package
        try:
            # Use pip show to get the real version
            show_out = subprocess.check_output([sys.executable, '-m', 'pip', 'show', package_name]).decode('utf-8')
            ver_match = re.search(r'^Version:\s*(.*)$', show_out, re.MULTILINE)
            if ver_match:
                version = ver_match.group(1).strip()
                clean_reqs.append(f"{package_name}=={version}")
            else:
                # Fallback: just package name
                clean_reqs.append(package_name)
        except:
            clean_reqs.append(package_name)
    else:
        # Standard line (package==version) or editable (-e ...)
        if line.startswith('-e'):
            continue # Skip editable git links usually, or keep if user wants "truly everything"
                     # But usually -e is a local path which breaks.
        clean_reqs.append(line)

# Sort for neatness
clean_reqs.sort()

# Manually verify criticals are present (just in case)
criticals = ['torch', 'tensorflow', 'mediapipe', 'scikit-learn', 'PySide6', 'box2d-py', 'gymnasium']
current_set = {r.split('==')[0].lower() for r in clean_reqs}

for crit in criticals:
    if crit.lower() not in current_set:
        # Try finding it with diverse casing or just append it
        print(f"Warning: {crit} not found in pip freeze. Appending generic.")
        clean_reqs.append(crit)

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(clean_reqs))

print(f"Success! Wrote {len(clean_reqs)} packages to requirements.txt")
