# -*- coding: utf-8 -*-
# https://confluence.sberbank.ru/pages/viewpage.action?pageId=17454139301
# https://confluence.sberbank.ru/pages/viewpage.action?pageId=12447784417

import os

from langchain_community.chat_models.gigachat import GigaChat
from lib.llm.risk_chain import MEFChatOpenAIModule, ChatOpenAIConfig, SSLConfig

class myllm:
  def __init__(self, name):
    self.mod = name
   
    self.chat =''
    if self.mod == 'giga':
      base_dir = os.path.dirname(__file__)
      cert_path = os.path.join(base_dir, 'client_cert.pem')
      key_path = os.path.join(base_dir, 'client_key.pem')
      self.chat = GigaChat(
          base_url='https://gigachat-ift.sberdevices.delta.sbrf.ru/v1',
          cert_file=cert_path,
          key_file=key_path,
          model='GigaChat-Max',
          temperature=0.5,
          top_p=0.2,
          verify_ssl_certs=False,
          profanity_check=False,
          streaming=True
      )
    if self.mod == 'qwen':
      chat_openai_config = ChatOpenAIConfig(
          ssl_config=SSLConfig(
              cert=(
                  r"C:\Users\19459394\Documents\19459394\Documents\Python Projects\lib\llm\mef_client_cert.pem",
                  r"C:\Users\19459394\Documents\19459394\Documents\Python Projects\lib\llm\mef_client_key.pem",
              ),
              verify=False,
          ),  # type: ignore
          model_name="Qwen3-32B-AWQ",
          openai_api_base="https://sberorm-llm-deploy.ci03039946-eiftmefds-"
                          "vectorizer-service.apps.ift-mef-ds.delta.sbrf.ru/"
                          "sberorm-llm-deploy",
          max_retries=1,
      )
      self.chat = MEFChatOpenAIModule().provide_chat_openai_client( chat_openai_config)
