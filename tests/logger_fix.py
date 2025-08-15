# logger_fix.py
"""
Быстрое исправление LangGraphLogger
Добавляет недостающий метод log_workflow_step
"""

import sys
from pathlib import Path

def fix_langgraph_logger():
    """Исправление LangGraphLogger в logger.py"""
    print("🔧 Исправление LangGraphLogger...")
    
    logger_file = Path("src/utils/logger.py")
    
    if logger_file.exists():
        content = logger_file.read_text(encoding='utf-8')
        
        # Ищем класс LangGraphLogger
        if "class LangGraphLogger:" in content:
            # Ищем метод log_quality_check и добавляем log_workflow_step после него
            quality_check_method = '''    def log_quality_check(self, assessment_id: str, risk_type: str, quality_score: float, threshold: float):
        """Логирование проверки качества критиком"""
        bound_logger = self.logger.bind_context(assessment_id, "quality_check")
        status = "✅ пройдена" if quality_score >= threshold else "❌ не пройдена"
        bound_logger.info(f"🔍 Проверка качества {risk_type}: {quality_score:.1f}/{threshold} - {status}")'''
            
            new_method = '''    def log_quality_check(self, assessment_id: str, risk_type: str, quality_score: float, threshold: float):
        """Логирование проверки качества критиком"""
        bound_logger = self.logger.bind_context(assessment_id, "quality_check")
        status = "✅ пройдена" if quality_score >= threshold else "❌ не пройдена"
        bound_logger.info(f"🔍 Проверка качества {risk_type}: {quality_score:.1f}/{threshold} - {status}")
    
    def log_workflow_step(self, assessment_id: str, step_name: str, details: str = ""):
        """Логирование шага workflow"""
        bound_logger = self.logger.bind_context(assessment_id, "workflow")
        message = f"⚙️ Workflow шаг: {step_name}"
        if details:
            message += f" - {details}"
        bound_logger.info(message)'''
            
            if quality_check_method in content and "def log_workflow_step" not in content:
                content = content.replace(quality_check_method, new_method)
                logger_file.write_text(content, encoding='utf-8')
                print("✅ Метод log_workflow_step добавлен в LangGraphLogger")
                return True
            elif "def log_workflow_step" in content:
                print("✅ Метод log_workflow_step уже существует")
                return True
            else:
                print("⚠️ Не удалось найти место для добавления метода")
                return False
        else:
            print("❌ Не найден класс LangGraphLogger")
            return False
    else:
        print("❌ Файл logger.py не найден")
        return False

def main():
    """Главная функция исправления"""
    print("🚀 ИСПРАВЛЕНИЕ LOGGER")
    print("=" * 30)
    
    success = fix_langgraph_logger()
    
    if success:
        print("\n✅ ИСПРАВЛЕНИЕ ПРИМЕНЕНО!")
        print("🚀 Теперь можно запустить тест:")
        print("   python quick_test_workflow.py")
    else:
        print("\n❌ ОШИБКА ИСПРАВЛЕНИЯ")
        print("🔧 Требуется ручное исправление")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)