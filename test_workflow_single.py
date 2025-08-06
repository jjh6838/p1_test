#!/usr/bin/env python3
"""
Simple test script for 1 country only - fastest test
"""

import subprocess
import sys
import time
from pathlib import Path

def test_one_country():
    """Test workflow with 1 small country for quick validation"""
    
    start_time = time.time()
    test_country = "JAM"  # Jamaica
    output_dir = "test_outputs_single"
    
    print("🧪 Testing workflow with 1 country (fastest test)")
    print(f"Country: {test_country}")
    print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # # Step 1: Get country list
    # print("\n1. Getting country list...")
    # try:
    #     subprocess.run([sys.executable, "get_countries.py"], check=True)
    #     print("✅ Country list created")
    # except subprocess.CalledProcessError as e:
    #     print(f"❌ Failed: {e}")
    #     return False
    
    # Step 2: Test the country
    print(f"\n1. Testing {test_country}...")
    step_start = time.time()
    try:
        cmd = [sys.executable, "process_country_supply.py", test_country, "--output-dir", output_dir]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        step_time = time.time() - step_start
        print(f"✅ {test_country} processing succeeded ({step_time:.1f}s)")
        
        # Check if output file was created
        output_file = Path(output_dir) / f"supply_analysis_{test_country}.parquet"
        if output_file.exists():
            print(f"✅ Output file created: {output_file}")
        else:
            print(f"⚠️  Output file not found: {output_file}")
            
    except subprocess.CalledProcessError as e:
        step_time = time.time() - step_start
        print(f"❌ {test_country} failed ({step_time:.1f}s)")
        print(f"Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False
    
    # Step 3: Test combine function (even with 1 file)
    print("\n2. Testing combine function...")
    step_start = time.time()
    try:
        cmd = [sys.executable, "combine_global_results.py", "--input-dir", output_dir, "--output-file", f"{output_dir}/global_test.parquet"]
        subprocess.run(cmd, check=True, capture_output=True)
        step_time = time.time() - step_start
        print(f"✅ Combine function works ({step_time:.1f}s)")
    except subprocess.CalledProcessError as e:
        step_time = time.time() - step_start
        print(f"❌ Combining failed ({step_time:.1f}s): {e}")
        return False
    
    total_time = time.time() - start_time
    print(f"\n🎉 Single country test PASSED!")
    print(f"⏱️  Total execution time: {total_time:.1f} seconds")
    print(f"📁 Test outputs in: {output_dir}/")
    print(f"🚀 Ready to run full workflow: snakemake --cores 4 --use-conda")
    return True

if __name__ == "__main__":
    success = test_one_country()
    sys.exit(0 if success else 1)
