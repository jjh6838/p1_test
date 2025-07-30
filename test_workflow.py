#!/usr/bin/env python3
"""
Simple test script for 2 countries before running full Snakemake workflow
"""

import subprocess
import sys
from pathlib import Path

def test_two_countries():
    """Test workflow with 2 small countries"""
    
    test_countries = ["LKA", "JAM"]  # Small, reliable countries
    output_dir = "test_outputs"
    
    print("🧪 Testing workflow with 2 countries")
    print(f"Countries: {test_countries}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Step 1: Get country list
    print("\n1. Getting country list...")
    try:
        subprocess.run([sys.executable, "get_countries.py"], check=True)
        print("✅ Country list created")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Step 2: Test each country
    print("\n2. Testing countries...")
    for country in test_countries:
        print(f"   Processing {country}...")
        try:
            cmd = [sys.executable, "process_country_supply.py", country, "--output-dir", output_dir]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"   ✅ {country} succeeded")
        except subprocess.CalledProcessError:
            print(f"   ❌ {country} failed")
            return False
    
    # Step 3: Combine results
    print("\n3. Combining results...")
    try:
        cmd = [sys.executable, "combine_global_results.py", "--input-dir", output_dir]
        subprocess.run(cmd, check=True, capture_output=True)
        print("✅ Results combined")
    except subprocess.CalledProcessError:
        print("❌ Combining failed")
        return False
    
    print(f"\n🎉 Test PASSED! Ready for Snakemake")
    print(f"Run: snakemake --cores 4 --use-conda")
    return True

if __name__ == "__main__":
    success = test_two_countries()
    sys.exit(0 if success else 1)
