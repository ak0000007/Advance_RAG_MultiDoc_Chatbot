from src.llm.provider import create_llm

def test_google_api():
    try:
        print("Initializing Google Gemini model...")
        llm = create_llm(provider="google")
        
        print("Sending test message...")
        response = llm.invoke("Say 'API key is working!' and nothing else.")
        
        print("\nSuccess! Response received:")
        print(f"> {response.content}")
        
    except Exception as e:
        print("\nError: API key test failed.")
        print(f"Details: {str(e)}")

if __name__ == "__main__":
    test_google_api()
