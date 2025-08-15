from src.utils.llm_config_manager import get_llm_config_manager, get_base_url, get_model

# Проверка работы конфигуратора
manager = get_llm_config_manager()
print("Конфигурация LLM:")
print(manager.get_info())

# Проверка convenience функций
print(f"URL: {get_base_url()}")
print(f"Model: {get_model()}")