"""
Центральный менеджер конфигурации LLM.
Единая точка управления всеми настройками языковых моделей.
ОБНОВЛЕНО: Добавлена поддержка GigaChat и DeepSeek API
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Ищем .env файл в текущей директории и родительских
    env_loaded = load_dotenv(verbose=True)  # verbose=True покажет какой файл загружен

    if not env_loaded:
        # Пытаемся найти .env в родительских директориях
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file, verbose=True)
                print(f"📁 Загружен .env файл: {env_file}")
                break
        else:
            print("⚠️  .env файл не найден, используются системные переменные окружения")
    else:
        print("📁 .env файл успешно загружен")

except ImportError:
    print("⚠️  python-dotenv не установлен, используются системные переменные окружения")
    print("   Установите: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Ошибка загрузки .env файла: {e}")

class LLMProvider(Enum):
    """Поддерживаемые провайдеры LLM"""
    LM_STUDIO = "lm_studio"
    GIGACHAT = "gigachat"
    DEEPSEEK = "deepseek"

@dataclass
class LLMConfig:
    """Конфигурация LLM клиента"""
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int
    retry_delay: float
    provider: LLMProvider = LLMProvider.LM_STUDIO
    api_key: Optional[str] = None  # Added for DeepSeek
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    top_p: float = 0.2
    verify_ssl_certs: bool = False
    profanity_check: bool = False
    streaming: bool = True

    @classmethod
    def from_manager(cls, **overrides) -> 'LLMConfig':
        """Создание конфигурации из центрального менеджера с возможностью переопределения"""
        global config_manager

        if config_manager is None or config_manager._config is None:
            temp_manager = LLMConfigManager()
            base_config = temp_manager.get_config()
        else:
            base_config = config_manager.get_config()

        return cls(
            base_url=overrides.get('base_url', base_config.base_url),
            model=overrides.get('model', base_config.model),
            temperature=overrides.get('temperature', base_config.temperature),
            max_tokens=overrides.get('max_tokens', base_config.max_tokens),
            timeout=overrides.get('timeout', base_config.timeout),
            max_retries=overrides.get('max_retries', base_config.max_retries),
            retry_delay=overrides.get('retry_delay', base_config.retry_delay),
            provider=overrides.get('provider', base_config.provider),
            api_key=overrides.get('api_key', base_config.api_key),
            cert_file=overrides.get('cert_file', base_config.cert_file),
            key_file=overrides.get('key_file', base_config.key_file),
            top_p=overrides.get('top_p', base_config.top_p),
            verify_ssl_certs=overrides.get('verify_ssl_certs', base_config.verify_ssl_certs),
            profanity_check=overrides.get('profanity_check', base_config.profanity_check),
            streaming=overrides.get('streaming', base_config.streaming)
        )

    @classmethod
    def create_default(cls) -> 'LLMConfig':
        """Создание конфигурации по умолчанию (fallback для обратной совместимости)"""
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234"),
            model=os.getenv("LLM_MODEL", "qwen3-4b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("MAX_RETRY_COUNT", "3")),
            retry_delay=float(os.getenv("LLM_RETRY_DELAY", "1.0")),
            provider=LLMProvider.LM_STUDIO
        )

    @classmethod
    def from_env(cls, **overrides) -> 'LLMConfig':
        """Альтернативный метод создания конфигурации напрямую из переменных окружения"""
        provider_str = os.getenv("LLM_PROVIDER", "lm_studio").lower()
        provider_mapping = {
            "lm_studio": LLMProvider.LM_STUDIO,
            "gigachat": LLMProvider.GIGACHAT,
            "deepseek": LLMProvider.DEEPSEEK
        }
        provider = provider_mapping.get(provider_str, LLMProvider.LM_STUDIO)

        if provider == LLMProvider.GIGACHAT:
            base_url = os.getenv("GIGACHAT_BASE_URL", "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1")
            model = os.getenv("GIGACHAT_MODEL", "GigaChat-Max")
            cert_path = os.getenv("GIGACHAT_CERT_PATH", "lib/llm/client_cert.pem")
            key_path = os.getenv("GIGACHAT_KEY_PATH", "lib/llm/client_key.pem")

            if not os.path.isabs(cert_path):
                cert_path = os.path.join(os.getcwd(), cert_path)
            if not os.path.isabs(key_path):
                key_path = os.path.join(os.getcwd(), key_path)

            cert_file = cert_path
            key_file = key_path
            top_p = float(os.getenv("GIGACHAT_TOP_P", "0.2"))
            verify_ssl_certs = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true"
            profanity_check = os.getenv("GIGACHAT_PROFANITY_CHECK", "false").lower() == "true"
            streaming = os.getenv("GIGACHAT_STREAMING", "true").lower() == "true"
            api_key = None
        elif provider == LLMProvider.DEEPSEEK:
            base_url = os.getenv("LLM_BASE_URL", "https://api.proxyapi.ru/deepseek")
            model = os.getenv("LLM_MODEL", "deepseek-chat")
            cert_file = None
            key_file = None
            top_p = float(os.getenv("LLM_TOP_P", "0.2"))
            verify_ssl_certs = os.getenv("LLM_VERIFY_SSL", "true").lower() == "true"
            profanity_check = False
            streaming = os.getenv("LLM_STREAMING", "true").lower() == "true"
            api_key = os.getenv("LLM_DeepSeek_API_KEY")
        else:
            base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
            model = os.getenv("LLM_MODEL", "qwen3-4b")
            cert_file = None
            key_file = None
            top_p = float(os.getenv("LLM_TOP_P", "0.2"))
            verify_ssl_certs = False
            profanity_check = False
            streaming = True
            api_key = None

        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        max_retries = int(os.getenv("MAX_RETRY_COUNT", "3"))
        retry_delay = float(os.getenv("LLM_RETRY_DELAY", "1.0"))

        return cls(
            base_url=overrides.get('base_url', base_url),
            model=overrides.get('model', model),
            temperature=overrides.get('temperature', temperature),
            max_tokens=overrides.get('max_tokens', max_tokens),
            timeout=overrides.get('timeout', timeout),
            max_retries=overrides.get('max_retries', max_retries),
            retry_delay=overrides.get('retry_delay', retry_delay),
            provider=overrides.get('provider', provider),
            api_key=overrides.get('api_key', api_key),
            cert_file=overrides.get('cert_file', cert_file),
            key_file=overrides.get('key_file', key_file),
            top_p=overrides.get('top_p', top_p),
            verify_ssl_certs=overrides.get('verify_ssl_certs', verify_ssl_certs),
            profanity_check=overrides.get('profanity_check', profanity_check),
            streaming=overrides.get('streaming', streaming)
        )

class LLMConfigManager:
    """
    Singleton менеджер конфигурации LLM.
    Управляет всеми настройками языковых моделей из одного места.
    ОБНОВЛЕНО: Поддержка множественных провайдеров включая DeepSeek
    """

    _instance: Optional['LLMConfigManager'] = None
    _config: Optional[LLMConfig] = None

    def __new__(cls) -> 'LLMConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """Загрузка конфигурации из переменных окружения"""
        provider_str = os.getenv("LLM_PROVIDER", "lm_studio").lower()
        provider = self._parse_provider(provider_str)

        if provider == LLMProvider.GIGACHAT:
            self._config = self._load_gigachat_config()
        elif provider == LLMProvider.DEEPSEEK:
            self._config = self._load_deepseek_config()
        else:
            self._config = self._load_lm_studio_config()

    def _parse_provider(self, provider_str: str) -> LLMProvider:
        """Парсинг провайдера из строки"""
        provider_mapping = {
            "lm_studio": LLMProvider.LM_STUDIO,
            "gigachat": LLMProvider.GIGACHAT,
            "deepseek": LLMProvider.DEEPSEEK
        }
        return provider_mapping.get(provider_str, LLMProvider.LM_STUDIO)

    def _load_gigachat_config(self) -> LLMConfig:
        """Загрузка конфигурации для GigaChat"""
        base_url = os.getenv("GIGACHAT_BASE_URL", "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1")
        model = os.getenv("GIGACHAT_MODEL", "GigaChat-Max")
        cert_path = os.getenv("GIGACHAT_CERT_PATH", "lib/llm/client_cert.pem")
        key_path = os.getenv("GIGACHAT_KEY_PATH", "lib/llm/client_key.pem")

        if not os.path.isabs(cert_path):
            cert_path = os.path.join(os.getcwd(), cert_path)
        if not os.path.isabs(key_path):
            key_path = os.path.join(os.getcwd(), key_path)

        if not (os.path.exists(cert_path) and os.path.exists(key_path)):
            raise FileNotFoundError(
                f"GigaChat SSL-файлы не найдены!\n"
                f"Cert: {cert_path}\n"
                f"Key: {key_path}\n"
                f"Проверьте пути в .env файле"
            )

        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        max_retries = int(os.getenv("MAX_RETRY_COUNT", "3"))
        retry_delay = float(os.getenv("LLM_RETRY_DELAY", "1.0"))

        return LLMConfig(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            provider=LLMProvider.GIGACHAT,
            cert_file=cert_path,
            key_file=key_path,
            top_p=float(os.getenv("GIGACHAT_TOP_P", "0.2")),
            verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true",
            profanity_check=os.getenv("GIGACHAT_PROFANITY_CHECK", "false").lower() == "true",
            streaming=os.getenv("GIGACHAT_STREAMING", "true").lower() == "true"
        )

    def _load_deepseek_config(self) -> LLMConfig:
        """Загрузка конфигурации для DeepSeek"""
        base_url = os.getenv("LLM_BASE_URL", "https://api.proxyapi.ru/deepseek")
        model = os.getenv("LLM_MODEL", "deepseek-chat")
        api_key = os.getenv("LLM_DeepSeek_API_KEY")

        if not api_key:
            raise ValueError("DeepSeek API key not found in environment variable LLM_DeepSeek_API_KEY")

        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        max_retries = int(os.getenv("MAX_RETRY_COUNT", "3"))
        retry_delay = float(os.getenv("LLM_RETRY_DELAY", "1.0"))

        return LLMConfig(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            provider=LLMProvider.DEEPSEEK,
            api_key=api_key,
            top_p=float(os.getenv("LLM_TOP_P", "0.2")),
            verify_ssl_certs=os.getenv("LLM_VERIFY_SSL", "true").lower() == "true",
            streaming=os.getenv("LLM_STREAMING", "true").lower() == "true"
        )

    def _load_lm_studio_config(self) -> LLMConfig:
        """Загрузка конфигурации для LM Studio"""
        base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
        model = os.getenv("LLM_MODEL", "qwen3-4b")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        max_retries = int(os.getenv("MAX_RETRY_COUNT", "3"))
        retry_delay = float(os.getenv("LLM_RETRY_DELAY", "1.0"))

        provider = self._detect_provider(base_url)

        return LLMConfig(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            provider=provider
        )

    def _detect_provider(self, base_url: str) -> LLMProvider:
        """Автоматическое определение провайдера по URL"""
        if "127.0.0.1" in base_url or "localhost" in base_url:
            return LLMProvider.LM_STUDIO
        elif "gigachat" in base_url.lower() or "sber" in base_url.lower():
            return LLMProvider.GIGACHAT
        elif "deepseek" in base_url.lower() or "proxyapi" in base_url.lower():
            return LLMProvider.DEEPSEEK
        else:
            return LLMProvider.LM_STUDIO

    def get_config(self) -> LLMConfig:
        """Получение полной конфигурации"""
        return self._config

    def get_base_url(self) -> str:
        """Получение базового URL для LLM API"""
        return self._config.base_url

    def get_model(self) -> str:
        """Получение названия модели"""
        return self._config.model

    def get_temperature(self) -> float:
        """Получение температуры для генерации"""
        return self._config.temperature

    def get_max_tokens(self) -> int:
        """Получение максимального количества токенов"""
        return self._config.max_tokens

    def get_timeout(self) -> int:
        """Получение таймаута запроса в секундах"""
        return self._config.timeout

    def get_max_retries(self) -> int:
        """Получение максимального количества повторных попыток"""
        return self._config.max_retries

    def get_retry_delay(self) -> float:
        """Получение задержки между повторными попытками"""
        return self._config.retry_delay

    def get_provider(self) -> LLMProvider:
        """Получение текущего провайдера"""
        return self._config.provider

    def get_api_key(self) -> Optional[str]:
        """Получение API ключа"""
        return self._config.api_key

    def get_quality_threshold(self) -> float:
        """Получение порога качества для оценок"""
        return float(os.getenv("QUALITY_THRESHOLD", "7.0"))

    def get_cert_file(self) -> Optional[str]:
        """Получение пути к файлу сертификата"""
        return self._config.cert_file

    def get_key_file(self) -> Optional[str]:
        """Получение пути к файлу ключа"""
        return self._config.key_file

    def get_top_p(self) -> float:
        """Получение параметра top_p"""
        return self._config.top_p

    def get_verify_ssl_certs(self) -> bool:
        """Получение настройки проверки SSL сертификатов"""
        return self._config.verify_ssl_certs

    def get_profanity_check(self) -> bool:
        """Получение настройки проверки ненормативной лексики"""
        return self._config.profanity_check

    def get_streaming(self) -> bool:
        """Получение настройки потокового режима"""
        return self._config.streaming

    def is_gigachat(self) -> bool:
        """Проверка, используется ли GigaChat"""
        return self._config.provider == LLMProvider.GIGACHAT

    def is_lm_studio(self) -> bool:
        """Проверка, используется ли LM Studio"""
        return self._config.provider == LLMProvider.LM_STUDIO

    def is_deepseek(self) -> bool:
        """Проверка, используется ли DeepSeek"""
        return self._config.provider == LLMProvider.DEEPSEEK

    def create_agent_config_dict(self, **overrides) -> Dict[str, Any]:
        """Создание словаря конфигурации для агентов с возможностью переопределения"""
        config = {
            "llm_base_url": self.get_base_url(),
            "llm_model": self.get_model(),
            "temperature": self.get_temperature(),
            "max_retries": self.get_max_retries(),
            "timeout_seconds": self.get_timeout(),
            "api_key": self.get_api_key()
        }

        if self.is_gigachat():
            config.update({
                "cert_file": self.get_cert_file(),
                "key_file": self.get_key_file(),
                "top_p": self.get_top_p(),
                "verify_ssl_certs": self.get_verify_ssl_certs(),
                "profanity_check": self.get_profanity_check(),
                "streaming": self.get_streaming()
            })

        config.update(overrides)
        return config

    def create_llm_client_config_dict(self, **overrides) -> Dict[str, Any]:
        """Создание словаря конфигурации для LLM клиента с возможностью переопределения"""
        config = {
            "base_url": self.get_base_url(),
            "model": self.get_model(),
            "temperature": self.get_temperature(),
            "api_key": self.get_api_key()
        }

        if self.is_gigachat():
            config.update({
                "cert_file": self.get_cert_file(),
                "key_file": self.get_key_file(),
                "top_p": self.get_top_p(),
                "verify_ssl_certs": self.get_verify_ssl_certs(),
                "profanity_check": self.get_profanity_check(),
                "streaming": self.get_streaming()
            })

        config.update(overrides)
        return config

    def reload_config(self) -> None:
        """Перезагрузка конфигурации из переменных окружения"""
        self._config = None
        self._load_config()

    def set_provider_config(self, provider: LLMProvider, config_updates: Dict[str, Any]) -> None:
        """Обновление конфигурации для конкретного провайдера"""
        for key, value in config_updates.items():
            os.environ[key] = str(value)

        self.reload_config()

    def get_info(self) -> Dict[str, Any]:
        """Получение информации о текущей конфигурации для отладки"""
        info = {
            "provider": self._config.provider.value,
            "base_url": self._config.base_url,
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "timeout": self._config.timeout,
            "max_retries": self._config.max_retries,
            "quality_threshold": self.get_quality_threshold(),
            "api_key": "******" if self._config.api_key else None
        }

        if self.is_gigachat():
            info.update({
                "cert_file": self._config.cert_file,
                "key_file": self._config.key_file,
                "top_p": self._config.top_p,
                "streaming": self._config.streaming,
                "verify_ssl_certs": self._config.verify_ssl_certs,
                "profanity_check": self._config.profanity_check
            })

        return info

# Глобальный экземпляр менеджера конфигурации
config_manager = LLMConfigManager()

def get_llm_config_manager() -> LLMConfigManager:
    """Получение глобального экземпляра менеджера конфигурации"""
    return config_manager

def get_base_url() -> str:
    """Быстрый доступ к базовому URL"""
    return config_manager.get_base_url()

def get_model() -> str:
    """Быстрый доступ к модели"""
    return config_manager.get_model()

def get_temperature() -> float:
    """Быстрый доступ к температуре"""
    return config_manager.get_temperature()

def get_llm_config() -> LLMConfig:
    """Быстрый доступ к полной конфигурации"""
    return config_manager.get_config()

def is_gigachat() -> bool:
    """Быстрая проверка, используется ли GigaChat"""
    return config_manager.is_gigachat()

def is_deepseek() -> bool:
    """Быстрая проверка, используется ли DeepSeek"""
    return config_manager.is_deepseek()

def force_reload_config():
    """Принудительная перезагрузка конфигурации (для тестов и переключения провайдеров)"""
    global config_manager
    if config_manager._instance is not None:
        config_manager._instance._config = None
        config_manager._instance._load_config()

    try:
        from . import llm_client
        llm_client._global_client = None
    except ImportError:
        pass

def reset_config_manager():
    """Полный сброс менеджера конфигурации (для тестов)"""
    global config_manager
    LLMConfigManager._instance = None
    config_manager = LLMConfigManager()

def print_env_diagnosis():
    """Диагностика переменных окружения"""
    import os

    print("\n🔍 ДИАГНОСТИКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ:")
    required_vars = ["LLM_PROVIDER"]
    gigachat_vars = ["GIGACHAT_CERT_PATH", "GIGACHAT_KEY_PATH", "GIGACHAT_BASE_URL", "GIGACHAT_MODEL"]
    deepseek_vars = ["LLM_DeepSeek_API_KEY"]

    for var in required_vars + gigachat_vars + deepseek_vars:
        value = os.getenv(var)
        status = "✅" if value else "❌"
        print(f"   {status} {var}: {value if value else 'НЕ УСТАНОВЛЕНА'}")

    if os.getenv("LLM_PROVIDER", "").lower() == "gigachat":
        cert_path = os.getenv("GIGACHAT_CERT_PATH", "")
        key_path = os.getenv("GIGACHAT_KEY_PATH", "")

        print(f"\n🔒 ПРОВЕРКА СЕРТИФИКАТОВ:")
        for name, path in [("CERT", cert_path), ("KEY", key_path)]:
            if path:
                if not os.path.isabs(path):
                    path = os.path.join(os.getcwd(), path)
                exists = os.path.exists(path)
                status = "✅" if exists else "❌"
                print(f"   {status} {name}: {path} ({'найден' if exists else 'НЕ НАЙДЕН'})")
            else:
                print(f"   ❌ {name}: путь не указан")