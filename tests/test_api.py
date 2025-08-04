"""
Created By: Anushuya Baidya
Date: 8/2/25
"""

import requests

def test_api():
    url = "http://localhost:5002"
    tests = [
        {"desc": "organic coffee shop", "expect": "success"},
        {"desc": "adult website", "expect": "blocked"},
        {"desc": "", "expect": "error"}
    ]

    print("Testing API...")
    for i, test in enumerate(tests, 1):
        print(f"\nTest {i}: {test['desc'] or 'empty'}")

        try:
            response = requests.post(f"{url}/generate", json={"business_description": test["desc"]}, timeout=5)
            result = response.json()

            print(f"Status: {result['status']} (expected: {test['expect']})")

            if result['status'] == 'success':
                domains = [s['domain'] for s in result['suggestions']]
                print(f"Domains: {domains}")
                print(result)

        except requests.exceptions.ConnectionError:
            print("❌ API not running")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()