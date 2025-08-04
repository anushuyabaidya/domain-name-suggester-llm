"""
Created By: Anushuya Baidya
Date: 8/2/25
"""


import requests

def test_api():
    """Test the domain API"""

    url = "http://localhost:5002"

    print("🧪 Testing Domain API...")

    tests = [
        {"desc": "organic coffee shop", "expect": "success"},
        {"desc": "AI tech startup", "expect": "success"},
        {"desc": "adult website", "expect": "blocked"},
        {"desc": "", "expect": "error"}
    ]

    for i, test in enumerate(tests, 1):
        print(f"\nTest {i}: {test['desc'] or 'empty'}")

        try:
            response = requests.post(
                f"{url}/generate",
                json={"business_description": test["desc"]},
                timeout=10
            )

            result = response.json()
            print(f"Status: {result['status']}")

            if result['status'] == 'success':
                domains = [s['domain'] for s in result['suggestions']]
                print(f"Domains: {domains}")
            else:
                print(f"Message: {result.get('message', 'No message')}")

        except requests.exceptions.ConnectionError:
            print("❌ API not running - start with: python domain_api.py")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_api()

