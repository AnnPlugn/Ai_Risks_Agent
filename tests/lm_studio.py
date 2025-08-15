import requests
import json

# Базовый URL для LM Studio
BASE_URL = "http://127.0.0.1:1234/v1"

def test_connection():
    """Проверка подключения к LM Studio"""
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            print("Подключение успешно! Доступные модели:")
            print(json.dumps(response.json(), indent=2))
            # Проверка наличия модели qwen/qwen3-8b
            models = response.json().get("data", [])
            if any(model["id"] == "qwen/qwen3-8b" for model in models):
                print("Модель qwen/qwen3-8b найдена!")
                return True
            else:
                print("Ошибка: Модель qwen/qwen3-8b не найдена.")
                return False
        else:
            print(f"Ошибка подключения: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("Ошибка: Не удалось подключиться к LM Studio. Проверьте, запущен ли сервер.")
        return False

def test_chat_completion():
    """Тестовый запрос к модели qwen/qwen3-8b"""
    if not test_connection():
        return

    url = f"{BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "qwen/qwen3-8b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, is the qwen/qwen3-8b model working?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            print("Ответ от модели qwen/qwen3-8b:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Ошибка запроса: {response.status_code} - {response.text}")
    except requests.ConnectionError:
        print("Ошибка: Не удалось выполнить запрос. Проверьте сервер LM Studio.")

if __name__ == "__main__":
    test_chat_completion()