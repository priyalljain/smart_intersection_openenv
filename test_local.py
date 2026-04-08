"""
Test the local OpenEnv server
Run AFTER: python -m uvicorn server.app:app --reload --port 8000
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_reset():
    """Test /reset endpoint"""
    print("[1] Testing /reset endpoint...")
    try:
        resp = requests.post(f"{BASE_URL}/reset", json={})
        print(f"  Status: {resp.status_code}")
        
        if resp.status_code != 200:
            print(f"  ❌ FAILED: Expected 200, got {resp.status_code}")
            print(f"  Response: {resp.text}")
            return False
        
        obs = resp.json()["observation"]
        print(f"  ✅ Reset successful")
        print(f"     - Phase: {obs['phase']}")
        print(f"     - Total Queue: {obs['total_queue_length']}")
        print(f"     - Emergencies: {obs['active_emergencies']}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def test_step():
    """Test /step endpoint"""
    print("\n[2] Testing /step endpoint...")
    try:
        # First reset
        requests.post(f"{BASE_URL}/reset", json={})
        
        # Then step
        resp = requests.post(
            f"{BASE_URL}/step",
            json={"action": {"phase": "ns"}}
        )
        print(f"  Status: {resp.status_code}")
        
        if resp.status_code != 200:
            print(f"  ❌ FAILED: Expected 200, got {resp.status_code}")
            print(f"  Response: {resp.text}")
            return False
        
        result = resp.json()
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        
        print(f"  ✅ Step successful")
        print(f"     - Reward: {reward:.6f}")
        print(f"     - Done: {done}")
        print(f"     - Phase: {obs['phase']}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def test_docs():
    """Test /docs endpoint (Swagger UI)"""
    print("\n[3] Testing /docs endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/docs")
        print(f"  Status: {resp.status_code}")
        
        if resp.status_code != 200:
            print(f"  ❌ FAILED: Expected 200, got {resp.status_code}")
            return False
        
        print(f"  ✅ Swagger UI available")
        print(f"     Open in browser: {BASE_URL}/docs")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def test_episode():
    """Test full episode"""
    print("\n[4] Running full 20-step episode...")
    try:
        # Reset
        requests.post(f"{BASE_URL}/reset", json={})
        
        rewards = []
        for step in range(20):
            phase = "ns" if step % 2 == 0 else "ew"
            resp = requests.post(
                f"{BASE_URL}/step",
                json={"action": {"phase": phase}}
            )
            
            if resp.status_code != 200:
                print(f"  ❌ Step {step+1} FAILED")
                return False
            
            reward = resp.json()["reward"]
            rewards.append(reward)
            
            if (step + 1) % 5 == 0:
                print(f"  Step {step+1}: reward={reward:.6f}, avg={sum(rewards)/len(rewards):.6f}")
        
        avg_reward = sum(rewards) / len(rewards)
        print(f"  ✅ Episode complete")
        print(f"     - Total steps: 20")
        print(f"     - Average reward: {avg_reward:.6f}")
        print(f"     - Total reward: {sum(rewards):.6f}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("OPENENV LOCAL SERVER TESTS")
    print("=" * 60)
    print(f"\nTesting server at: {BASE_URL}\n")
    
    results = []
    results.append(("Reset", test_reset()))
    results.append(("Step", test_step()))
    results.append(("Docs", test_docs()))
    results.append(("Episode", test_episode()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
    
    all_passed = all(p for _, p in results)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nNext steps:")
        print("  1. Build Docker: docker build -t traffic-control:latest .")
        print("  2. Run Docker: docker run -p 8000:8000 traffic-control:latest")
        print("  3. Push to HF: openenv push --repo-id your-username/traffic-control-env")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nTroubleshooting:")
        print("  1. Is uvicorn running? python -m uvicorn server.app:app --reload --port 8000")
        print("  2. Check server logs for errors")
        print("  3. Verify env.py is using env_CORRECT.py")
