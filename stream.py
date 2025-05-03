import json
import requests
import sseclient  # Ensure sseclient-py is installed

url = "http://127.0.0.1:5000/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

history = [{"role": "user", "content": "Hello, how are you?"}]
data = {
    "mode": "instruct",
    "stream": True,
    "messages": history
}

stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
client = sseclient.SSEClient(stream_response)

assistant_message = ''
for event in client.events():
    print(f"Raw event data: {event.data}")
    try:
        payload = json.loads(event.data)
        if 'choices' in payload and len(payload['choices']) > 0:
            delta = payload['choices'][0]['delta']
            if 'content' in delta:
                chunk = delta['content']
                assistant_message += chunk
                print(f"Parsed chunk: {chunk}")
    except Exception as e:
        print(f"Error parsing event data: {e}")
        print(f"Event data that caused error: {event.data}")
print("\nAssistant's full message:")
print(assistant_message)
