import requests
import json

# --- API Configuration ---

API_BASE_URL = "http://localhost:8000/"
CLASSIFY_ENDPOINT = "/classify/"

# --- Text to Classify ---

text_to_send = "cats eat fish"

request_payload = {
    "text": text_to_send
}

headers = {
    "Content-Type": "application/json"
}

# --- Make the API Call ---
try:
    print(f"Sending request to: {API_BASE_URL}{CLASSIFY_ENDPOINT}")
    print(f"Text being sent: '{text_to_send}'")

    response = requests.post(
        url=f"{API_BASE_URL}{CLASSIFY_ENDPOINT}",
        headers=headers,
        data=json.dumps(request_payload) # Convert Python dict to JSON string
    )

    response.raise_for_status()

    # --- Process the Response ---
    api_response = response.json()

    print("\n--- API Response ---")
    print(json.dumps(api_response, indent=2))

    # --- Interpret and Print Results ---
    prediction = api_response.get("prediction")
    negative_prob = api_response.get("negative_probability")
    positive_prob = api_response.get("positive_probability")

    if prediction is not None:
        predicted_label = "Positive" if prediction == 1 else "Negative"
        print(f"\nPredicted Sentiment: {predicted_label}")
    if negative_prob is not None and positive_prob is not None:
        print(f"Negative Probability: {negative_prob:.4f}")
        print(f"Positive Probability: {positive_prob:.4f}")

except requests.exceptions.ConnectionError as e:
    print(f"\nError: Could not connect to the API at {API_BASE_URL}.")
    print("Please ensure:")
    print("  1. The API is running on the host machine.")
    print("  2. 'YOUR_MACHINE_IP_ADDRESS' is correctly set to the host's actual IP.")
    print("  3. There are no firewall rules blocking port 8000 on the host.")
    print(f"Connection Error Details: {e}")
except requests.exceptions.HTTPError as e:
    print(f"\nError: HTTP Request failed with status code {e.response.status_code}")
    print(f"Response from API: {e.response.text}")
    print(f"HTTP Error Details: {e}")
except json.JSONDecodeError as e:
    print(f"\nError: Could not decode JSON response from API.")
    print(f"Raw response text: {response.text}")
    print(f"JSON Decode Error Details: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")