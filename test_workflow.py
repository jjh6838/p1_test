#!/usr/bin/env python3
"""
Simple test script for 2 countries before running full Snakemake workflow
"""

import subprocess
import sys
import time
from pathlib import Path

def test_two_countries():
    """Test workflow with 2 small countries"""
    
    start_time = time.time()
    test_countries = ["LKA", "JAM"]  # Small, reliable countries (TUV, NRU)
    output_dir = "test_outputs"
    
    print("🧪 Testing workflow with 2 countries")
    print(f"Countries: {test_countries}")
    print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Step 1: Get country list
    print("\n1. Getting country list...")
    step_start = time.time()
    try:
        subprocess.run([sys.executable, "get_countries.py"], check=True)
        step_time = time.time() - step_start
        print(f"✅ Country list created ({step_time:.1f}s)")
    except subprocess.CalledProcessError as e:
        step_time = time.time() - step_start
        print(f"❌ Failed ({step_time:.1f}s): {e}")
        return False
    
    # Step 2: Test each country
    print("\n2. Testing countries...")
    countries_start = time.time()
    for i, country in enumerate(test_countries, 1):
        print(f"   Processing {country} ({i}/{len(test_countries)})...")
        country_start = time.time()
        try:
            cmd = [sys.executable, "process_country_supply.py", country, "--output-dir", output_dir]
            subprocess.run(cmd, check=True, capture_output=True)
            country_time = time.time() - country_start
            print(f"   ✅ {country} succeeded ({country_time:.1f}s)")
        except subprocess.CalledProcessError:
            country_time = time.time() - country_start
            print(f"   ❌ {country} failed ({country_time:.1f}s)")
            return False
    
    countries_time = time.time() - countries_start
    print(f"Countries processing completed in {countries_time:.1f}s")
    
    # Step 3: Combine results
    print("\n3. Combining results...")
    step_start = time.time()
    try:
        cmd = [sys.executable, "combine_global_results.py", "--input-dir", output_dir]
        subprocess.run(cmd, check=True, capture_output=True)
        step_time = time.time() - step_start
        print(f"✅ Results combined ({step_time:.1f}s)")
    except subprocess.CalledProcessError:
        step_time = time.time() - step_start
        print(f"❌ Combining failed ({step_time:.1f}s)")
        return False
    
    total_time = time.time() - start_time
    print(f"\n🎉 Test PASSED!")
    print(f"⏱️  Total execution time: {total_time:.1f} seconds")
    print(f"📊 Average per country: {countries_time/len(test_countries):.1f}s")
    print(f"🚀 Ready for Snakemake: snakemake --cores 4 --use-conda")
    return True

if __name__ == "__main__":
    success = test_two_countries()
    sys.exit(0 if success else 1)
