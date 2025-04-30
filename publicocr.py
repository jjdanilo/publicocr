# -*- coding: utf-8 -*-
# --- Imports ---
import sys
import time
import threading
import os
import re
from concurrent.futures import ThreadPoolExecutor, Future
import requests # Para LibreTranslate
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QLabel, QMessageBox, QInputDialog, QRubberBand,
                             QComboBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint, QRect
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QScreen
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
import paddle
import mss
import pygetwindow as gw
import traceback
from difflib import SequenceMatcher
import string
import configparser
from pathlib import Path

# --- API Dependencies ---
try:
    import deepl
except ImportError:
    print("AVISO: Biblioteca 'deepl' não encontrada. Instale com 'pip install deepl' para usar a API DeepL.")
    deepl = None
try:
    from google.cloud import translate_v2 as google_translate
except ImportError:
    print("AVISO: Biblioteca 'google-cloud-translate' não encontrada. Instale com 'pip install google-cloud-translate' para usar a API Google.")
    google_translate = None

# --- Serviços de Tradução ---
LIBRETRANSLATE = "LibreTranslate"
DEEPL = "DeepL"
GOOGLE = "Google Translate"
OCR_SERVICE = "PaddleOCR" # Identificador para o serviço de OCR

# --- Configuração ---
# SOURCE_LANG_OCR e TARGET_LANG_TRANSLATE removidos, serão gerenciados pela GUI/config
LIBRETRANSLATE_URL = "http://127.0.0.1:5000"
REQUESTS_TIMEOUT = 10
CAPTURE_INTERVAL_MS = 100
MIN_WINDOW_WIDTH = 100
MIN_WINDOW_HEIGHT = 100
NUM_TRANSLATION_THREADS = 2
OCR_SIMILARITY_THRESHOLD = 0.96

# --- Configuração de Persistência ---
CONFIG_FILE_NAME = ".ocr_translator_config.ini"
CONFIG_SECTION = "Settings"
CONFIG_KEY_DEEPL = "deepl_api_key"
CONFIG_KEY_SERVICE = "last_translation_service"
CONFIG_KEY_SOURCE_LANG = "source_language" # NOVO
CONFIG_KEY_TARGET_LANG = "target_language" # NOVO

# --- Definição de Idiomas Suportados ---
# Chave: Nome amigável para exibição na GUI
# Valor: Dicionário com códigos:
#    'internal': Código usado internamente na aplicação (consistente)
#    'ocr': Código esperado pelo PaddleOCR (VERIFIQUE A DOCUMENTAÇÃO DO PADDLEOCR!)
#    'deepl': Código esperado pelo DeepL API (ex: 'EN-US', 'PT-BR')
#    'google': Código esperado pelo Google Translate API (ISO 639-1)
#    'libre': Código esperado pelo LibreTranslate (geralmente ISO 639-1)
#
# !!! IMPORTANTE: Verifique se os idiomas de origem ('ocr') são realmente suportados
#     pelo modelo do PaddleOCR que você está usando (ex: 'ch', 'en', 'fr', 'german', 'korean', 'japan') !!!
#     Nem todos os idiomas listados aqui podem funcionar com o OCR.
SUPPORTED_LANGUAGES = {
    "Inglês": {
        "internal": "en",
        "ocr": "en",       # PaddleOCR suporta 'en'
        "deepl": "EN",
        "google": "en",
        "libre": "en"
    },
    "Português (Brasil)": {
        "internal": "pt-br",
        "ocr": "pt",       # PaddleOCR pode usar 'pt' (verificar modelo) ou talvez precise de um modelo específico
        "deepl": "pt-br",
        "google": "pt",
        "libre": "pt-br"
    },
    "Espanhol": {
        "internal": "es",
        "ocr": "es",       # PaddleOCR suporta 'es'
        "deepl": "ES",
        "google": "es",
        "libre": "es"
    },
    "Francês": {
        "internal": "fr",
        "ocr": "fr",       # PaddleOCR suporta 'fr'
        "deepl": "FR",
        "google": "fr",
        "libre": "fr"
    },
    "Alemão": {
        "internal": "de",
        "ocr": "german",   # PaddleOCR usa 'german'
        "deepl": "DE",
        "google": "de",
        "libre": "de"
    },
    "Italiano": {
        "internal": "it",
        "ocr": "it",       # PaddleOCR suporta 'it'
        "deepl": "IT",
        "google": "it",
        "libre": "it"
    },
    "Russo": {
        "internal": "ru",
        "ocr": "ru",       # PaddleOCR suporta 'ru'
        "deepl": "RU",
        "google": "ru",
        "libre": "ru"
    },
    "Japonês": {
        "internal": "ja",
        "ocr": "japan",    # PaddleOCR usa 'japan'
        "deepl": "JA",
        "google": "ja",
        "libre": "ja"
    },
    "Chinês (Simplificado)": {
        "internal": "zh-cn",
        "ocr": "ch",       # PaddleOCR usa 'ch' para chinês simplificado/tradicional
        "deepl": "ZH",     # DeepL usa 'ZH' para chinês (simplificado)
        "google": "zh-CN", # Google usa códigos específicos
        "libre": "zh"      # LibreTranslate geralmente usa 'zh'
    },
    "Coreano": {
        "internal": "ko",
        "ocr": "korean",   # PaddleOCR usa 'korean'
        "deepl": "KO",
        "google": "ko",
        "libre": "ko"
    },
    # Adicione mais idiomas se necessário, sempre verificando os códigos
}

# Idiomas padrão se a configuração falhar
DEFAULT_SOURCE_LANG_INTERNAL = "en"
DEFAULT_TARGET_LANG_INTERNAL = "pt-br"
DEFAULT_TRANSLATION_SERVICE = LIBRETRANSLATE

def get_config_path():
    """Retorna o caminho completo para o arquivo de configuração."""
    return Path.home() / CONFIG_FILE_NAME

def load_config():
    """Carrega a configuração do arquivo INI."""
    config_path = get_config_path()
    config = configparser.ConfigParser()
    settings = {
        CONFIG_KEY_DEEPL: None,
        CONFIG_KEY_SERVICE: DEFAULT_TRANSLATION_SERVICE,
        CONFIG_KEY_SOURCE_LANG: DEFAULT_SOURCE_LANG_INTERNAL,
        CONFIG_KEY_TARGET_LANG: DEFAULT_TARGET_LANG_INTERNAL
    }

    if config_path.exists():
        try:
            config.read(config_path)
            if CONFIG_SECTION in config:
                settings[CONFIG_KEY_DEEPL] = config[CONFIG_SECTION].get(CONFIG_KEY_DEEPL, None)
                settings[CONFIG_KEY_SERVICE] = config[CONFIG_SECTION].get(CONFIG_KEY_SERVICE, DEFAULT_TRANSLATION_SERVICE)
                settings[CONFIG_KEY_SOURCE_LANG] = config[CONFIG_SECTION].get(CONFIG_KEY_SOURCE_LANG, DEFAULT_SOURCE_LANG_INTERNAL)
                settings[CONFIG_KEY_TARGET_LANG] = config[CONFIG_SECTION].get(CONFIG_KEY_TARGET_LANG, DEFAULT_TARGET_LANG_INTERNAL)

                if settings[CONFIG_KEY_DEEPL] == "None" or settings[CONFIG_KEY_DEEPL] == "":
                    settings[CONFIG_KEY_DEEPL] = None
                print(f"Configuração carregada de {config_path}: Serviço='{settings[CONFIG_KEY_SERVICE]}', "
                      f"Origem='{settings[CONFIG_KEY_SOURCE_LANG]}', Destino='{settings[CONFIG_KEY_TARGET_LANG]}', "
                      f"Chave DeepL {'presente' if settings[CONFIG_KEY_DEEPL] else 'ausente'}.")
        except configparser.Error as e:
            print(f"Erro ao ler arquivo de configuração {config_path}: {e}")
        except Exception as e:
             print(f"Erro inesperado ao carregar configuração: {e}")

    # Validações pós-carregamento
    available_services = [LIBRETRANSLATE]
    if deepl: available_services.append(DEEPL)
    if google_translate: available_services.append(GOOGLE)
    if settings[CONFIG_KEY_SERVICE] not in available_services:
        print(f"Aviso: Serviço salvo '{settings[CONFIG_KEY_SERVICE]}' não está disponível. Usando '{DEFAULT_TRANSLATION_SERVICE}'.")
        settings[CONFIG_KEY_SERVICE] = DEFAULT_TRANSLATION_SERVICE

    valid_internal_langs = [lang_data["internal"] for lang_data in SUPPORTED_LANGUAGES.values()]
    if settings[CONFIG_KEY_SOURCE_LANG] not in valid_internal_langs:
         print(f"Aviso: Idioma de origem salvo '{settings[CONFIG_KEY_SOURCE_LANG]}' inválido. Usando '{DEFAULT_SOURCE_LANG_INTERNAL}'.")
         settings[CONFIG_KEY_SOURCE_LANG] = DEFAULT_SOURCE_LANG_INTERNAL
    if settings[CONFIG_KEY_TARGET_LANG] not in valid_internal_langs:
         print(f"Aviso: Idioma de destino salvo '{settings[CONFIG_KEY_TARGET_LANG]}' inválido. Usando '{DEFAULT_TARGET_LANG_INTERNAL}'.")
         settings[CONFIG_KEY_TARGET_LANG] = DEFAULT_TARGET_LANG_INTERNAL

    return settings

def save_config(deepl_key, last_service, source_lang, target_lang):
    """Salva a configuração no arquivo INI."""
    config_path = get_config_path()
    config = configparser.ConfigParser()
    config[CONFIG_SECTION] = {}
    config[CONFIG_SECTION][CONFIG_KEY_DEEPL] = str(deepl_key) if deepl_key else ""
    config[CONFIG_SECTION][CONFIG_KEY_SERVICE] = last_service
    config[CONFIG_SECTION][CONFIG_KEY_SOURCE_LANG] = source_lang
    config[CONFIG_SECTION][CONFIG_KEY_TARGET_LANG] = target_lang

    try:
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        print(f"Configuração salva em {config_path}: Serviço='{last_service}', Origem='{source_lang}', Destino='{target_lang}', Chave DeepL {'presente' if deepl_key else 'ausente'}.")
    except IOError as e:
        print(f"Erro ao salvar arquivo de configuração {config_path}: {e}")
    except Exception as e:
        print(f"Erro inesperado ao salvar configuração: {e}")

def get_lang_code(internal_code, service_name):
    """
    Obtém o código de idioma específico para o serviço (OCR, DeepL, Google, Libre).
    Retorna o código interno como fallback se não encontrar mapeamento.
    """
    for lang_name, lang_data in SUPPORTED_LANGUAGES.items():
        if lang_data["internal"] == internal_code:
            # Mapeia o nome do serviço para a chave no dicionário lang_data
            service_key_map = {
                OCR_SERVICE: "ocr",
                DEEPL: "deepl",
                GOOGLE: "google",
                LIBRETRANSLATE: "libre"
            }
            service_key = service_key_map.get(service_name)
            if service_key and service_key in lang_data:
                code = lang_data[service_key]
                if code: # Verifica se o código não é None ou vazio
                    #print(f"Debug: Mapeado '{internal_code}' para '{code}' para o serviço '{service_name}'")
                    return code
                else:
                    print(f"Aviso: Código de idioma VAZIO para '{internal_code}' no serviço '{service_name}' no SUPPORTED_LANGUAGES.")
            else:
                 print(f"Aviso: Mapeamento de serviço '{service_name}' ou chave '{service_key}' não encontrado para '{internal_code}' no SUPPORTED_LANGUAGES.")
            # Fallback se o código específico não foi encontrado ou estava vazio
            print(f"Usando código interno '{internal_code}' como fallback para o serviço '{service_name}'.")
            return internal_code # Retorna o código interno como fallback

    # Fallback final se nem o código interno foi encontrado (não deveria acontecer se a validação funcionar)
    print(f"ERRO: Código interno '{internal_code}' não encontrado em SUPPORTED_LANGUAGES. Usando código como está.")
    return internal_code


# --- Widgets de Overlay e Legenda (sem alterações) ---
class AreaRefinementOverlay(QWidget):
    area_refined = pyqtSignal(dict)
    def __init__(self, parent_geometry):
        super().__init__()
        self.parent_geometry = parent_geometry
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground, False)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 50);")
        self.setCursor(Qt.CrossCursor)
        self.setGeometry(self.parent_geometry)
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        self.selection_rect_local = QRect()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.origin = event.pos(); self.rubber_band.setGeometry(QRect(self.origin, QPoint())); self.rubber_band.show(); event.accept()
        else: event.ignore()
    def mouseMoveEvent(self, event):
        if not self.origin.isNull(): self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized()); event.accept()
        else: event.ignore()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self.origin.isNull():
            self.selection_rect_local = self.rubber_band.geometry(); self.rubber_band.hide(); self.close()
            if self.selection_rect_local.width() > 10 and self.selection_rect_local.height() > 10:
                global_x = self.parent_geometry.x() + self.selection_rect_local.x(); global_y = self.parent_geometry.y() + self.selection_rect_local.y()
                bbox = {'left': global_x, 'top': global_y, 'width': self.selection_rect_local.width(), 'height': self.selection_rect_local.height()}
                self.area_refined.emit(bbox)
            else: print("Área refinada muito pequena."); self.area_refined.emit({})
            self.origin = QPoint(); event.accept()
        else: event.ignore()
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape: self.close(); self.area_refined.emit({})

class ScreenAreaSelectorOverlay(QWidget):
    screen_area_selected = pyqtSignal(QRect)
    def __init__(self):
        super().__init__()
        screen_geometry = QApplication.primaryScreen().geometry()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground); self.setAttribute(Qt.WA_NoSystemBackground, False)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 30);"); self.setCursor(Qt.CrossCursor)
        self.setGeometry(screen_geometry); self.rubber_band = QRubberBand(QRubberBand.Rectangle, self); self.origin = QPoint()
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.origin = event.pos(); self.rubber_band.setGeometry(QRect(self.origin, QPoint())); self.rubber_band.show(); event.accept()
        else: event.ignore()
    def mouseMoveEvent(self, event):
        if not self.origin.isNull(): self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized()); event.accept()
        else: event.ignore()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self.origin.isNull():
            selection_rect_global = self.rubber_band.geometry(); self.rubber_band.hide(); self.close()
            if selection_rect_global.width() > 20 and selection_rect_global.height() > 10: self.screen_area_selected.emit(selection_rect_global)
            else: print("Área de legenda selecionada muito pequena."); self.screen_area_selected.emit(QRect())
            self.origin = QPoint(); event.accept()
        else: event.ignore()
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape: self.close(); self.screen_area_selected.emit(QRect())

class SubtitleWindow(QWidget):
    def __init__(self, initial_geometry=QRect(10, 10, 300, 50)):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setStyleSheet("QWidget { background-color: black; color: yellow; border: 1px solid yellow; padding: 4px; }")
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(0, 0, 0, 0)
        self.text_label = QLabel("Aguardando tradução..."); self.text_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.text_label.setWordWrap(True); self.text_label.setStyleSheet("border: none; background-color: transparent;")
        self.layout.addWidget(self.text_label); self.setLayout(self.layout); self.setGeometry(initial_geometry)
        self._drag_start_position = None
    def update_text(self, text): self.text_label.setText(text)
    def update_geometry(self, rect): self.setGeometry(rect)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self._drag_start_position = event.globalPos() - self.frameGeometry().topLeft(); event.accept()
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_start_position: self.move(event.globalPos() - self._drag_start_position); event.accept()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton: self._drag_start_position = None; event.accept()

# --- Classe Worker (Modificada para aceitar idiomas e chave DeepL) ---
class Worker(QObject):
    update_text_signal = pyqtSignal(str, str)
    status_update_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    STATUS_PREFIX_TO_FILTER = "Status: "

    def __init__(self, capture_bbox,
                 ocr_lang_internal,   # Código interno do idioma de origem
                 target_lang_internal,# Código interno do idioma de destino
                 interval_ms,
                 translation_service,
                 libretranslate_url,
                 deepl_api_key,
                 num_threads):
        super().__init__()
        self.capture_bbox = capture_bbox
        self.ocr_lang_internal = ocr_lang_internal     # Armazena código interno OCR
        self.target_lang_internal = target_lang_internal # Armazena código interno Tradução
        self.interval_ms = interval_ms
        self.translation_service = translation_service
        self.libretranslate_url = libretranslate_url.rstrip('/')
        self.deepl_api_key = deepl_api_key
        self.num_threads = num_threads
        self._is_running = False
        self._stop_event = threading.Event()
        self.last_accepted_ocr_text = ""
        self.translation_executor = None
        self.translation_request_id = 0
        self.last_displayed_request_id = -1
        self.result_lock = threading.Lock()
        self.pending_originals = {}
        self.reader = None
        self.sct = None
        self.deepl_translator = None
        self.google_translator = None

    def _normalize_for_similarity(self, text):
        if not text: return ""
        normalized = text.lower()
        normalized = normalized.translate(str.maketrans('', '', string.punctuation))
        normalized = ' '.join(normalized.split()).strip()
        return normalized

    def _filter_status_messages(self, text):
        if not text: return ""
        lines = text.splitlines()
        filtered_lines = [line for line in lines if not line.strip().startswith(self.STATUS_PREFIX_TO_FILTER)]
        return '\n'.join(filtered_lines).strip()

    def _initialize_resources(self):
        # --- Inicializa OCR com idioma selecionado ---
        ocr_code_for_paddle = get_lang_code(self.ocr_lang_internal, OCR_SERVICE)
        self.status_update_signal.emit(f"Inicializando OCR (PaddleOCR - Idioma: {ocr_code_for_paddle})...")
        try:
            use_gpu = paddle.is_compiled_with_cuda()
            # *** IMPORTANTE: PaddleOCR pode lançar erro aqui se o idioma não for suportado ***
            self.reader = PaddleOCR(use_angle_cls=True, lang=ocr_code_for_paddle, use_gpu=use_gpu, show_log=False)
            print(f"PaddleOCR inicializado para '{ocr_code_for_paddle}' com {'GPU' if use_gpu else 'CPU'}.")
            if not use_gpu: print("Aviso: PaddleOCR pode ser mais rápido com GPU.")
        except Exception as e_ocr:
            # Tenta dar uma mensagem mais útil se for erro de idioma
            if "support lang" in str(e_ocr).lower() or "not found" in str(e_ocr).lower():
                 error_msg = f"Falha ao inicializar PaddleOCR: Idioma '{ocr_code_for_paddle}' (derivado de '{self.ocr_lang_internal}') não é suportado pelo modelo PaddleOCR instalado. Verifique os idiomas suportados."
                 self.error_signal.emit(error_msg)
            else:
                 self.error_signal.emit(f"Falha ao inicializar PaddleOCR ({ocr_code_for_paddle}): {e_ocr}")
            traceback.print_exc()
            return False

        # --- Inicializa Captura de Tela ---
        try:
            self.sct = mss.mss()
            print(f"Capturador de tela (mss) inicializado. Área: {self.capture_bbox}")
        except Exception as e_mss:
            self.error_signal.emit(f"Falha ao inicializar MSS: {e_mss}")
            return False

        # --- Inicialização Condicional dos Clientes de Tradução ---
        self.status_update_signal.emit(f"Configurando tradutor: {self.translation_service}...")
        # (Lógica de inicialização dos clientes DeepL, Google, LibreTranslate permanece a mesma)
        if self.translation_service == DEEPL:
            if not deepl: self.error_signal.emit("Erro: Biblioteca DeepL não instalada."); return False
            if not self.deepl_api_key: self.error_signal.emit("Erro: Chave API DeepL não fornecida."); return False
            try:
                self.deepl_translator = deepl.Translator(self.deepl_api_key)
                self.deepl_translator.get_usage()
                print("Cliente DeepL inicializado e autenticado.")
            except deepl.AuthorizationException: self.error_signal.emit("Falha na autenticação DeepL: Chave API inválida."); traceback.print_exc(); return False
            except Exception as e_deepl: self.error_signal.emit(f"Falha ao inicializar/autenticar DeepL: {e_deepl}"); traceback.print_exc(); return False
        elif self.translation_service == GOOGLE:
            if not google_translate: self.error_signal.emit("Erro: Biblioteca Google Translate não instalada."); return False
            try:
                if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'): print("AVISO: Variável GOOGLE_APPLICATION_CREDENTIALS não definida.")
                self.google_translator = google_translate.Client()
                langs = self.google_translator.get_languages()
                print(f"Cliente Google Translate inicializado. {len(langs)} idiomas suportados.")
            except Exception as e_google: self.error_signal.emit(f"Falha ao inicializar Google Translate: {e_google}"); traceback.print_exc(); return False
        elif self.translation_service == LIBRETRANSLATE:
            print(f"Usando servidor LibreTranslate em: {self.libretranslate_url}")
            try:
                response = requests.get(f"{self.libretranslate_url}/languages", timeout=5)
                if response.status_code != 200: print(f"Aviso: Falha ao conectar ao servidor LibreTranslate ({response.status_code}).")
                else: print("Conexão com LibreTranslate OK.")
            except requests.exceptions.RequestException as req_err: print(f"Aviso: Falha ao conectar ao servidor LibreTranslate ({req_err}).")
        else:
            self.error_signal.emit(f"Erro: Serviço de tradução desconhecido: {self.translation_service}"); return False

        self.translation_executor = ThreadPoolExecutor(max_workers=self.num_threads, thread_name_prefix='TranslatorAPI')
        print(f"Executor de chamadas API inicializado com {self.num_threads} threads.")
        return True

    # --- Funções de Tradução Específicas (usam get_lang_code) ---

    def _translate_libre(self, text_to_translate, request_id, original_ocr_text):
        translate_url = f"{self.libretranslate_url}/translate"
        # Usa os códigos internos para buscar os códigos específicos do LibreTranslate
        source_lang_code = get_lang_code(self.ocr_lang_internal, LIBRETRANSLATE)
        target_lang_code = get_lang_code(self.target_lang_internal, LIBRETRANSLATE)
        payload = {'q': text_to_translate, 'source': source_lang_code, 'target': target_lang_code, 'format': 'text'}
        headers = {'Content-Type': 'application/json'}
        start_time = time.time()
        try:
            response = requests.post(translate_url, json=payload, headers=headers, timeout=REQUESTS_TIMEOUT)
            duration = time.time() - start_time
            print(f"[Libre ID:{request_id}] {source_lang_code}->{target_lang_code}, Status: {response.status_code}, Tempo: {duration:.3f}s")
            if response.status_code == 200:
                try:
                    translated_text = response.json().get('translatedText', '').strip()
                    return original_ocr_text, translated_text or "[API Retornou Vazio]", request_id
                except requests.exceptions.JSONDecodeError: return original_ocr_text, "[Erro API: JSON Inválido]", request_id
            else:
                error_detail = f"Status {response.status_code}"
                try: error_detail += f" - {response.text[:100]}"
                except Exception: pass
                # Verifica erro comum de idioma não suportado
                if "Language not supported" in response.text:
                     error_detail = f"Idioma não suportado ({source_lang_code} ou {target_lang_code})"
                return original_ocr_text, f"[Erro API Libre: {error_detail}]", request_id
        except requests.exceptions.RequestException as req_err:
            return original_ocr_text, f"[Erro API Libre: {type(req_err).__name__}]", request_id

    def _translate_deepl(self, text_to_translate, request_id, original_ocr_text):
        if not self.deepl_translator: return original_ocr_text, "[Erro Interno: Cliente DeepL não inicializado]", request_id
        # Usa os códigos internos para buscar os códigos específicos do DeepL
        source_lang_code = get_lang_code(self.ocr_lang_internal, DEEPL)
        target_lang_code = get_lang_code(self.target_lang_internal, DEEPL)
        start_time = time.time()
        try:
            # DeepL pode detectar source, mas especificar pode ser mais robusto
            # A função translate_text retorna um objeto TextResult (ou lista deles,
            # mas aqui esperamos um único objeto pois a entrada é uma string única)
            result = self.deepl_translator.translate_text(
                text_to_translate,
                source_lang=source_lang_code, # Pode ser None se quiser autodetectar
                target_lang=target_lang_code
            )
            duration = time.time() - start_time
            translated_text = result.text.strip()

            # ----- CORREÇÃO APLICADA AQUI -----
            # O atributo correto é detected_source_lang
            detected_source = result.detected_source_lang
            # -----------------------------------

            print(f"[DeepL ID:{request_id}] {source_lang_code or 'auto'}({detected_source})->{target_lang_code}, Tempo: {duration:.3f}s, Chars: {len(text_to_translate)}")
            return original_ocr_text, translated_text or "[API Retornou Vazio]", request_id
        except deepl.AuthorizationException as auth_err: return original_ocr_text, f"[Erro API DeepL: Chave Inválida]", request_id
        except deepl.QuotaExceededException: return original_ocr_text, f"[Erro API DeepL: Cota Excedida]", request_id
        except deepl.DeepLException as dl_err:
             # Verifica erro de idioma
             if "target_lang" in str(dl_err) or "source_lang" in str(dl_err) or "value is not supported" in str(dl_err):
                 return original_ocr_text, f"[Erro API DeepL: Idioma não suportado ({source_lang_code}/{target_lang_code})]", request_id
             print(f"[DeepL ID:{request_id}] Erro DeepL: {dl_err}")
             traceback.print_exc() # Adicionado para mais detalhes no console em outros erros DeepL
             return original_ocr_text, f"[Erro API DeepL: {dl_err}]", request_id
        except Exception as e:
            print(f"[DeepL ID:{request_id}] Erro inesperado: {e}")
            traceback.print_exc() # Adicionado para mais detalhes no console
            return original_ocr_text, f"[Erro DeepL Inesperado: {type(e).__name__}]", request_id
        if not self.deepl_translator: return original_ocr_text, "[Erro Interno: Cliente DeepL não inicializado]", request_id
        # Usa os códigos internos para buscar os códigos específicos do DeepL
        source_lang_code = get_lang_code(self.ocr_lang_internal, DEEPL)
        target_lang_code = get_lang_code(self.target_lang_internal, DEEPL)
        start_time = time.time()
        try:
            # DeepL pode detectar source, mas especificar pode ser mais robusto
            result = self.deepl_translator.translate_text(
                text_to_translate,
                source_lang=source_lang_code, # Pode ser None se quiser autodetectar
                target_lang=target_lang_code
            )
            duration = time.time() - start_time
            translated_text = result.text.strip()
            detected_source = result.detected_source_language
            print(f"[DeepL ID:{request_id}] {source_lang_code or 'auto'}({detected_source})->{target_lang_code}, Tempo: {duration:.3f}s, Chars: {len(text_to_translate)}")
            return original_ocr_text, translated_text or "[API Retornou Vazio]", request_id
        except deepl.AuthorizationException as auth_err: return original_ocr_text, f"[Erro API DeepL: Chave Inválida]", request_id
        except deepl.QuotaExceededException: return original_ocr_text, f"[Erro API DeepL: Cota Excedida]", request_id
        except deepl.DeepLException as dl_err:
             # Verifica erro de idioma
             if "target_lang" in str(dl_err) or "source_lang" in str(dl_err) or "value is not supported" in str(dl_err):
                 return original_ocr_text, f"[Erro API DeepL: Idioma não suportado ({source_lang_code}/{target_lang_code})]", request_id
             print(f"[DeepL ID:{request_id}] Erro DeepL: {dl_err}")
             return original_ocr_text, f"[Erro API DeepL: {dl_err}]", request_id
        except Exception as e: print(f"[DeepL ID:{request_id}] Erro inesperado: {e}"); return original_ocr_text, f"[Erro DeepL Inesperado: {type(e).__name__}]", request_id

    def _translate_google(self, text_to_translate, request_id, original_ocr_text):
        if not self.google_translator: return original_ocr_text, "[Erro Interno: Cliente Google não inicializado]", request_id
        # Usa os códigos internos para buscar os códigos específicos do Google
        source_lang_code = get_lang_code(self.ocr_lang_internal, GOOGLE)
        target_lang_code = get_lang_code(self.target_lang_internal, GOOGLE)
        start_time = time.time()
        try:
            result = self.google_translator.translate(
                text_to_translate,
                target_language=target_lang_code,
                source_language=source_lang_code # Especificar ajuda a evitar erros
            )
            duration = time.time() - start_time
            translated_text = result['translatedText'].strip()
            import html
            translated_text = html.unescape(translated_text)
            detected_source = result.get('detectedSourceLanguage', 'N/A')
            print(f"[Google ID:{request_id}] {source_lang_code}->{target_lang_code} (Detectado: {detected_source}), Tempo: {duration:.3f}s")
            return original_ocr_text, translated_text or "[API Retornou Vazio]", request_id
        except Exception as e_google:
            print(f"[Google ID:{request_id}] Erro Google Translate: {e_google}")
            error_msg = str(e_google)
            if "provide credential" in error_msg.lower() or "permission denied" in error_msg.lower() or "403" in error_msg: error_msg = "Credenciais Inválidas/Ausentes"
            elif "invalid language code" in error_msg.lower() or "target language is invalid" in error_msg.lower(): error_msg = f"Idioma não suportado ({source_lang_code}/{target_lang_code})"
            elif hasattr(e_google, 'message'): error_msg = e_google.message
            return original_ocr_text, f"[Erro API Google: {error_msg[:100]}]", request_id

    # --- Função Dispatcher (_translate_task) ---
    # (Nenhuma alteração necessária aqui, já chama as funções corretas)
    def _translate_task(self, text_to_translate, request_id, original_ocr_text):
        try:
            if self.translation_service == LIBRETRANSLATE: return self._translate_libre(text_to_translate, request_id, original_ocr_text)
            elif self.translation_service == DEEPL: return self._translate_deepl(text_to_translate, request_id, original_ocr_text)
            elif self.translation_service == GOOGLE: return self._translate_google(text_to_translate, request_id, original_ocr_text)
            else: print(f"ERRO INTERNO: Serviço inválido: {self.translation_service}"); return original_ocr_text, "[Erro: Serviço Inválido]", request_id
        except Exception as e: error_msg = f"Erro Inesperado _translate_task ({self.translation_service}, ID:{request_id}): {type(e).__name__}"; print(f"{error_msg} - Detalhe: {e}"); traceback.print_exc(); return original_ocr_text, f"[{error_msg}]", request_id

    # --- Callback (_translation_done_callback) ---
    # (Nenhuma alteração necessária aqui)
    def _translation_done_callback(self, future: Future):
        if not self._is_running: return
        try:
            original_ocr, translated_text, request_id = future.result()
            self.pending_originals.pop(request_id, None)
            with self.result_lock:
                if request_id >= self.last_displayed_request_id:
                    self.last_displayed_request_id = request_id
                    is_api_error = translated_text.startswith(("[Erro API", "[Erro DeepL", "[Erro Google", "[Erro Interno", "[Erro:")) or translated_text == "[API Retornou Vazio]"
                    self.update_text_signal.emit(original_ocr, translated_text)
                    status_msg = f"Tradução ID {request_id} ({self.translation_service}) exibida." if not is_api_error else f"Exibido {translated_text} (ID: {request_id})"
                    self.status_update_signal.emit(status_msg)
                else:
                    print(f"Resultado DESCARTADO ID: {request_id} (Último exibido: {self.last_displayed_request_id})")
        except Exception as e:
            req_id_err = "desconhecido"
            try:
                 if future.done() and not future.cancelled(): _, _, req_id_err = future.result()
            except Exception as inner_e: print(f"Erro ao obter ID no callback: {inner_e}")
            error_msg = f"Erro callback/tarefa API ({self.translation_service}, ID:{req_id_err}): {type(e).__name__}: {e}"
            print(error_msg); traceback.print_exc()
            self.error_signal.emit(error_msg)
            if req_id_err != "desconhecido": self.pending_originals.pop(req_id_err, None)

    # --- Loop Principal (run) ---
    # (Nenhuma alteração significativa aqui, _initialize_resources já usa o idioma correto)
    def run(self):
        self._is_running = True; self._stop_event.clear(); self.last_displayed_request_id = -1; self.translation_request_id = 0; self.pending_originals.clear()
        self.last_accepted_ocr_text = ""

        if not isinstance(self.capture_bbox, dict) or not all(k in self.capture_bbox for k in ('top', 'left', 'width', 'height')) or self.capture_bbox['width'] <= 0 or self.capture_bbox['height'] <= 0:
            self.error_signal.emit(f"Área de captura final inválida: {self.capture_bbox}."); self._is_running = False; return

        if not self._initialize_resources(): # Inicializa OCR com idioma correto e APIs
             self._is_running = False; return

        last_capture_time = 0; self.status_update_signal.emit(f"Iniciando captura ({self.ocr_lang_internal} -> {self.target_lang_internal}). Tradutor: {self.translation_service}")
        while self._is_running and not self._stop_event.is_set():
            current_time = time.time()
            if current_time - last_capture_time < self.interval_ms / 1000.0: time.sleep(0.01); continue
            last_capture_time = current_time
            try:
                sct_img = self.sct.grab(self.capture_bbox)
                img_pil = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                img_np = np.array(img_pil)

                ocr_output = self.reader.ocr(img_np, cls=True)
                original_ocr_text_list = []
                if ocr_output and isinstance(ocr_output, list) and len(ocr_output) > 0 and ocr_output[0]:
                    for line_info in ocr_output[0]:
                        if line_info and isinstance(line_info, (list, tuple)) and len(line_info) >= 2:
                            text_part = line_info[1]
                            if text_part and isinstance(text_part, (list, tuple)) and len(text_part) > 0:
                                text = text_part[0]
                                if text: original_ocr_text_list.append(text)
                original_ocr_text = ' '.join(original_ocr_text_list).strip()

                if original_ocr_text:
                    text_to_process = self._filter_status_messages(original_ocr_text)
                    if text_to_process:
                        normalized_current = self._normalize_for_similarity(text_to_process)
                        normalized_last = self._normalize_for_similarity(self.last_accepted_ocr_text)
                        similarity_ratio = SequenceMatcher(None, normalized_last, normalized_current).ratio()

                        if similarity_ratio < OCR_SIMILARITY_THRESHOLD:
                            #print(f"OCR Mudou (Sim: {similarity_ratio:.2f}). Texto: '{text_to_process[:50]}...'")
                            self.last_accepted_ocr_text = text_to_process
                            self.translation_request_id += 1
                            current_request_id = self.translation_request_id
                            self.pending_originals[current_request_id] = original_ocr_text
                            self.status_update_signal.emit(f"Novo OCR (ID:{current_request_id}, Sim:{similarity_ratio:.2f}), submetendo ({self.translation_service})...")
                            future = self.translation_executor.submit(self._translate_task, text_to_process, current_request_id, original_ocr_text)
                            future.add_done_callback(self._translation_done_callback)

                            max_pending = self.num_threads * 3
                            if len(self.pending_originals) > max_pending:
                                try: oldest_id = min(self.pending_originals.keys()); self.pending_originals.pop(oldest_id, None)
                                except ValueError: pass
                        #else:
                        #    self.status_update_signal.emit(f"Capturando (OCR similar, Sim:{similarity_ratio:.2f})...")
                    #else:
                    #    self.status_update_signal.emit("Capturando (apenas status filtrado)...")
                #else:
                    #self.status_update_signal.emit("Capturando (sem texto detectado)...")
                    #if self.last_accepted_ocr_text:
                    #     print("OCR não detectou texto, resetando último texto aceito.")
                    #     self.last_accepted_ocr_text = ""

            except mss.ScreenShotError as sse: error_msg = f"Erro ao capturar área ({sse}). Janela/área inválida?"; print(error_msg); self.error_signal.emit(error_msg); time.sleep(2)
            except Exception as e: tb_str = traceback.format_exc(); error_msg = f"Erro Processamento OCR/Loop: {type(e).__name__}: {e}"; print(f"{error_msg}\nTraceback:\n{tb_str}"); self.error_signal.emit(error_msg); time.sleep(1)

        self._cleanup(); self.status_update_signal.emit("Captura parada.")

    # --- Stop e Cleanup ---
    # (Nenhuma alteração necessária aqui)
    def stop(self):
        if not self._is_running: return
        self.status_update_signal.emit("Parando captura e chamadas API...")
        self._is_running = False; self._stop_event.set()
        if self.translation_executor:
            print("Solicitando shutdown do executor de chamadas API...")
            cancel_futures = sys.version_info >= (3, 9)
            self.translation_executor.shutdown(wait=False, cancel_futures=cancel_futures)
            print("Shutdown solicitado.")

    def _cleanup(self):
        print("Iniciando limpeza de recursos do Worker...")
        if self.sct:
            try: self.sct.close(); print("Recurso MSS liberado.")
            except Exception as e: print(f"Erro ao fechar MSS: {e}")
            finally: self.sct = None
        if self.reader:
            try: del self.reader; self.reader = None; print("Recurso PaddleOCR Reader liberado.")
            except Exception as e: print(f"Erro ao deletar PaddleOCR Reader: {e}")
        if self.translation_executor:
             if not self._stop_event.is_set(): self.stop()
             self.translation_executor = None
        self.pending_originals.clear(); self.last_accepted_ocr_text = ""
        self.deepl_translator = None; self.google_translator = None
        print("Limpeza de recursos do Worker concluída.")

# --- Classe Principal da GUI (Modificada para gerenciar idiomas) ---
class OCRTranslatorApp(QWidget):
    def __init__(self, initial_config):
        super().__init__()
        self.worker_thread = None; self.worker = None; self.selected_window_obj = None; self.capture_bbox = None
        self.refinement_overlay = None; self.screen_selector_overlay = None; self.subtitle_window = None; self.subtitle_bbox = None

        # Carrega configuração inicial (incluindo idiomas)
        self.deepl_api_key = initial_config.get(CONFIG_KEY_DEEPL, None)
        self.selected_service = initial_config.get(CONFIG_KEY_SERVICE, DEFAULT_TRANSLATION_SERVICE)
        self.selected_source_lang = initial_config.get(CONFIG_KEY_SOURCE_LANG, DEFAULT_SOURCE_LANG_INTERNAL)
        self.selected_target_lang = initial_config.get(CONFIG_KEY_TARGET_LANG, DEFAULT_TARGET_LANG_INTERNAL)

        # Mapeamento reverso para encontrar nome amigável a partir do código interno
        self.internal_to_display_name = {v["internal"]: k for k, v in SUPPORTED_LANGUAGES.items()}

        self.initUI()

    def initUI(self):
        self.setWindowTitle(f'OCR Translator - Config') # Título mais curto
        self.setGeometry(100, 100, 750, 600) # Aumenta altura para idiomas
        main_layout = QVBoxLayout()
        top_controls_layout = QHBoxLayout()
        lang_layout = QHBoxLayout() # NOVO para idiomas
        service_layout = QHBoxLayout()
        action_buttons_layout = QHBoxLayout()
        text_layout = QHBoxLayout()

        # --- Controles Superiores (Seleção Janela/Área) ---
        self.select_window_button = QPushButton('1. Selecionar Janela'); self.select_window_button.clicked.connect(self.select_window_dialog)
        self.refine_area_button = QPushButton('2. Refinar Área OCR'); self.refine_area_button.clicked.connect(self.show_refinement_overlay); self.refine_area_button.setEnabled(False)
        self.select_subtitle_area_button = QPushButton('Definir Área Legenda'); self.select_subtitle_area_button.clicked.connect(self.select_subtitle_area); self.select_subtitle_area_button.setToolTip("Onde a legenda traduzida deve aparecer.")
        self.window_label = QLabel("Nenhuma janela selecionada"); self.window_label.setStyleSheet("font-style: italic; color: grey;")
        top_controls_layout.addWidget(self.select_window_button); top_controls_layout.addWidget(self.refine_area_button); top_controls_layout.addWidget(self.select_subtitle_area_button); top_controls_layout.addWidget(self.window_label, 1)

        # --- Seletores de Idioma ---
        self.source_lang_label = QLabel("Idioma Origem (OCR):")
        self.source_lang_selector = QComboBox()
        self.target_lang_label = QLabel("Idioma Destino (Tradução):")
        self.target_lang_selector = QComboBox()

        # Popula os ComboBoxes com os nomes amigáveis
        display_names = list(SUPPORTED_LANGUAGES.keys())
        self.source_lang_selector.addItems(display_names)
        self.target_lang_selector.addItems(display_names)

        # Define a seleção inicial baseada na config carregada
        initial_source_display = self.internal_to_display_name.get(self.selected_source_lang, display_names[0])
        initial_target_display = self.internal_to_display_name.get(self.selected_target_lang, display_names[1] if len(display_names)>1 else display_names[0])
        self.source_lang_selector.setCurrentText(initial_source_display)
        self.target_lang_selector.setCurrentText(initial_target_display)

        # Conecta os sinais
        self.source_lang_selector.currentIndexChanged.connect(self.on_source_lang_changed)
        self.target_lang_selector.currentIndexChanged.connect(self.on_target_lang_changed)

        lang_layout.addStretch()
        lang_layout.addWidget(self.source_lang_label)
        lang_layout.addWidget(self.source_lang_selector)
        lang_layout.addStretch()
        lang_layout.addWidget(self.target_lang_label)
        lang_layout.addWidget(self.target_lang_selector)
        lang_layout.addStretch()

        # --- Seletor de Serviço ---
        self.service_label = QLabel("Tradutor:")
        self.service_selector = QComboBox()
        self.available_services = [LIBRETRANSLATE]
        if deepl: self.available_services.append(DEEPL)
        if google_translate: self.available_services.append(GOOGLE)
        self.service_selector.addItems(self.available_services)

        if self.selected_service in self.available_services:
            try: self.service_selector.setCurrentIndex(self.available_services.index(self.selected_service))
            except ValueError: self.service_selector.setCurrentIndex(0); self.selected_service = self.service_selector.currentText()
        else: self.service_selector.setCurrentIndex(0); self.selected_service = self.service_selector.currentText()

        self.service_selector.currentIndexChanged.connect(self.on_service_changed)
        service_layout.addStretch()
        service_layout.addWidget(self.service_label)
        service_layout.addWidget(self.service_selector)
        service_layout.addStretch()

        # --- Botões de Ação ---
        self.start_button = QPushButton('3. Iniciar Captura'); self.start_button.clicked.connect(self.start_capture); self.start_button.setEnabled(False)
        self.stop_button = QPushButton('Parar Captura'); self.stop_button.clicked.connect(self.stop_capture); self.stop_button.setEnabled(False)
        action_buttons_layout.addStretch(); action_buttons_layout.addWidget(self.start_button); action_buttons_layout.addWidget(self.stop_button)

        # --- Áreas de Texto ---
        self.original_label = QLabel(f'Texto Original ({self.selected_source_lang}):'); self.original_text = QTextEdit(); self.original_text.setReadOnly(True); self.original_text.setMinimumWidth(350); self.original_text.setLineWrapMode(QTextEdit.NoWrap)
        self.translated_label = QLabel(f'Tradução ({self.selected_target_lang} via {self.selected_service}):'); self.translated_text = QTextEdit(); self.translated_text.setReadOnly(True); self.translated_text.setMinimumWidth(350)
        original_col_layout = QVBoxLayout(); original_col_layout.addWidget(self.original_label); original_col_layout.addWidget(self.original_text)
        translated_col_layout = QVBoxLayout(); translated_col_layout.addWidget(self.translated_label); translated_col_layout.addWidget(self.translated_text)
        text_layout.addLayout(original_col_layout); text_layout.addLayout(translated_col_layout)

        # --- Status ---
        self.status_label = QLabel('Status: Parado'); self.status_label.setStyleSheet("color: grey;"); self.status_label.setAlignment(Qt.AlignCenter)

        # --- Montagem Layout Principal ---
        main_layout.addLayout(top_controls_layout)
        main_layout.addLayout(lang_layout) # Adiciona layout de idiomas
        main_layout.addLayout(service_layout)
        main_layout.addLayout(action_buttons_layout)
        main_layout.addLayout(text_layout)
        main_layout.addWidget(self.status_label)
        self.setLayout(main_layout)

        QTimer.singleShot(100, self.check_initial_service_requirements)
        self.show()

    def get_internal_code_from_display(self, display_name):
        """Retorna o código interno correspondente ao nome amigável."""
        for name, data in SUPPORTED_LANGUAGES.items():
            if name == display_name:
                return data["internal"]
        print(f"ERRO: Nome amigável '{display_name}' não encontrado em SUPPORTED_LANGUAGES.")
        return None # Ou retornar um padrão?

    def on_source_lang_changed(self, index):
        """Chamado quando o idioma de origem (OCR) muda."""
        display_name = self.source_lang_selector.currentText()
        internal_code = self.get_internal_code_from_display(display_name)
        if internal_code:
            self.selected_source_lang = internal_code
            self.original_label.setText(f'Texto Original ({self.selected_source_lang}):')
            print(f"Idioma de origem (OCR) alterado para: {display_name} ({self.selected_source_lang})")
            self.save_current_config() # Salva imediatamente
            # Poderia adicionar verificação se PaddleOCR suporta aqui, mas é melhor na inicialização do Worker
        else:
             QMessageBox.warning(self, "Erro Interno", f"Não foi possível encontrar o código interno para '{display_name}'.")

    def on_target_lang_changed(self, index):
        """Chamado quando o idioma de destino (Tradução) muda."""
        display_name = self.target_lang_selector.currentText()
        internal_code = self.get_internal_code_from_display(display_name)
        if internal_code:
            self.selected_target_lang = internal_code
            self.translated_label.setText(f'Tradução ({self.selected_target_lang} via {self.selected_service}):')
            print(f"Idioma de destino (Tradução) alterado para: {display_name} ({self.selected_target_lang})")
            self.save_current_config() # Salva imediatamente
        else:
             QMessageBox.warning(self, "Erro Interno", f"Não foi possível encontrar o código interno para '{display_name}'.")

    def on_service_changed(self, index):
        """Chamado quando o serviço de tradução muda."""
        self.selected_service = self.service_selector.currentText()
        self.translated_label.setText(f'Tradução ({self.selected_target_lang} via {self.selected_service}):')
        print(f"Serviço de tradução alterado para: {self.selected_service}")
        self.save_current_config() # Salva imediatamente

        if self.selected_service == DEEPL:
            if not self.deepl_api_key: self.prompt_for_deepl_key(force_prompt=True)
        elif self.selected_service == GOOGLE:
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'): QMessageBox.warning(self, "Credenciais Google Ausentes", "Variável GOOGLE_APPLICATION_CREDENTIALS não definida. Tradução pode falhar.")

    def save_current_config(self):
        """Salva a configuração atual (chave, serviço, idiomas)."""
        save_config(self.deepl_api_key, self.selected_service, self.selected_source_lang, self.selected_target_lang)

    def check_initial_service_requirements(self):
        """Verifica requisitos do serviço selecionado ao iniciar."""
        if self.selected_service == DEEPL:
            if not self.deepl_api_key:
                QMessageBox.information(self, "Chave DeepL Necessária", "DeepL selecionado, mas sem chave API salva. Será solicitada ao iniciar.")
        elif self.selected_service == GOOGLE:
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                 QMessageBox.warning(self, "Credenciais Google Ausentes", "Google selecionado, mas GOOGLE_APPLICATION_CREDENTIALS não definida. Tradução pode falhar.")

    def prompt_for_deepl_key(self, force_prompt=True):
        """Solicita a chave API DeepL ao usuário e a salva."""
        if not deepl: QMessageBox.critical(self, "Erro", "Biblioteca DeepL não instalada."); return False
        current_key = self.deepl_api_key
        prompt_needed = force_prompt or not current_key
        if prompt_needed:
            text, ok = QInputDialog.getText(self, 'Chave API DeepL', 'Insira sua chave API DeepL:', text=current_key if current_key else "")
            if ok:
                if text and len(text) > 10:
                    self.deepl_api_key = text.strip(); print("Chave DeepL inserida."); self.save_current_config(); QMessageBox.information(self, "Chave Salva", "Chave API DeepL salva."); return True
                elif text: QMessageBox.warning(self, "Chave Inválida?", "Chave inserida parece inválida. Nenhuma chave salva."); self.deepl_api_key = None; self.save_current_config(); return False
                else: self.deepl_api_key = None; print("Chave DeepL removida."); self.save_current_config(); QMessageBox.information(self, "Chave Removida", "Chave API DeepL removida."); return False
            else: print("Entrada da chave DeepL cancelada."); return bool(self.deepl_api_key)
        else: return True

    # --- Métodos de Seleção de Janela/Área (sem alterações) ---
    def select_window_dialog(self):
        try:
            all_windows = gw.getAllWindows(); window_titles = []; self.window_map = {}
            active_window_hwnd = None
            try: active_window = gw.getActiveWindow(); active_window_hwnd = active_window._hWnd if active_window else None
            except: pass
            for window in all_windows:
                if window.title and window.width >= MIN_WINDOW_WIDTH and window.height >= MIN_WINDOW_HEIGHT:
                     is_active_translator_window = False; my_title = self.windowTitle()
                     if active_window_hwnd and hasattr(window,'_hWnd') and window._hWnd == active_window_hwnd and my_title in window.title: is_active_translator_window = True
                     elif my_title in window.title: is_active_translator_window = True
                     if not is_active_translator_window:
                         title = f"{window.title} ({window.width}x{window.height})"
                         if title not in self.window_map: window_titles.append(title); self.window_map[title] = window
            if not window_titles: QMessageBox.information(self, "Nenhuma Janela", "Não foram encontradas janelas adequadas."); return
            window_titles.sort(); item, ok = QInputDialog.getItem(self, "Selecionar Janela", "Escolha a janela para capturar o texto:", window_titles, 0, False)
            if ok and item:
                selected_window_obj = self.window_map.get(item)
                if selected_window_obj:
                    self.selected_window_obj = selected_window_obj
                    try:
                        geo = self.selected_window_obj.box
                        if geo is None or geo.left is None or geo.top is None or geo.width <= 0 or geo.height <= 0:
                            time.sleep(0.2); geo_alt = gw.getWindowGeometry(self.selected_window_obj.title)
                            if geo_alt and geo_alt.left is not None and geo_alt.top is not None and geo_alt.width > 0 and geo_alt.height > 0: geo = geo_alt
                            else: raise ValueError(f"Geometria inválida para '{item}'.")
                        self.capture_bbox = {'left': geo.left, 'top': geo.top, 'width': geo.width, 'height': geo.height}
                        self.window_label.setText(f"Janela: {self.selected_window_obj.title}. Refine a área!"); self.window_label.setStyleSheet("color: orange; font-weight: bold;")
                        self.refine_area_button.setEnabled(True); self.start_button.setEnabled(False); self.status_label.setText("Status: Janela selecionada. Refine a área de captura."); self.status_label.setStyleSheet("color: orange;")
                    except Exception as e: QMessageBox.warning(self, "Erro Geometria", f"Não foi possível obter geometria da janela '{item}': {e}"); self.selected_window_obj = None; self.capture_bbox = None; self.refine_area_button.setEnabled(False); self.start_button.setEnabled(False); self.window_label.setText("Erro geometria"); self.window_label.setStyleSheet("color: red;")
                else: QMessageBox.warning(self, "Erro Interno", "Objeto da janela não encontrado.")
            elif self.selected_window_obj is None: self.window_label.setText("Seleção cancelada"); self.window_label.setStyleSheet("font-style: italic; color: orange;")
        except Exception as e: QMessageBox.critical(self, "Erro Listar Janelas", f"Erro inesperado: {e}"); traceback.print_exc(); self.selected_window_obj = None; self.capture_bbox = None; self.refine_area_button.setEnabled(False); self.start_button.setEnabled(False); self.window_label.setText("Erro ao listar"); self.window_label.setStyleSheet("color: red;")

    def show_refinement_overlay(self):
        if not self.selected_window_obj: QMessageBox.warning(self, "Janela Necessária", "Selecione uma janela primeiro."); return
        try:
            try:
                if self.selected_window_obj.isActive == False: self.selected_window_obj.activate(); QApplication.processEvents(); time.sleep(0.2); QApplication.processEvents()
            except Exception as act_err: print(f"Aviso: Não ativar janela '{self.selected_window_obj.title}': {act_err}")
            geo = self.selected_window_obj.box
            if geo is None or geo.left is None or geo.top is None or geo.width <= 0 or geo.height <= 0:
                 time.sleep(0.2); geo_alt = gw.getWindowGeometry(self.selected_window_obj.title)
                 if geo_alt and geo_alt.left is not None and geo_alt.top is not None and geo_alt.width > 0 and geo_alt.height > 0: geo = geo_alt
                 else: QMessageBox.warning(self, "Erro", f"Tamanho/posição da janela '{self.selected_window_obj.title}' inválida."); return
            parent_rect = QRect(geo.left, geo.top, geo.width, geo.height)
            self.hide(); QTimer.singleShot(150, lambda: self._create_and_show_overlay(parent_rect))
        except Exception as e: self.show(); QMessageBox.warning(self, "Erro Refinamento", f"Erro ao preparar refinamento: {e}"); traceback.print_exc()

    def _create_and_show_overlay(self, parent_rect):
        self.refinement_overlay = AreaRefinementOverlay(parent_rect); self.refinement_overlay.area_refined.connect(self.on_area_refined); self.refinement_overlay.show(); self.refinement_overlay.activateWindow(); self.refinement_overlay.raise_()

    def on_area_refined(self, bbox):
        self.show(); self.activateWindow(); self.raise_(); self.refinement_overlay = None
        if bbox and isinstance(bbox, dict) and all(k in bbox for k in ['left', 'top', 'width', 'height']):
            self.capture_bbox = bbox; self.window_label.setText(f"Área OCR: [{bbox['width']}x{bbox['height']}] @ ({bbox['left']},{bbox['top']})"); self.window_label.setStyleSheet("color: green;")
            self.start_button.setEnabled(True); self.status_label.setText("Status: Área OCR definida. Pronto."); self.status_label.setStyleSheet("color: green;"); print(f"Área OCR refinada: {self.capture_bbox}")
        else:
            if self.selected_window_obj: self.window_label.setText(f"Janela: {self.selected_window_obj.title}. Refinamento cancelado."); self.window_label.setStyleSheet("color: orange; font-weight: bold;")
            else: self.window_label.setText("Refinamento cancelado."); self.window_label.setStyleSheet("color: orange;")
            self.start_button.setEnabled(False); self.capture_bbox = None

    def select_subtitle_area(self):
        print("Selecione a área para a legenda..."); self.hide(); QTimer.singleShot(150, self._create_and_show_screen_selector)

    def _create_and_show_screen_selector(self):
        self.screen_selector_overlay = ScreenAreaSelectorOverlay(); self.screen_selector_overlay.screen_area_selected.connect(self.on_subtitle_area_selected); self.screen_selector_overlay.show(); self.screen_selector_overlay.activateWindow(); self.screen_selector_overlay.raise_()

    def on_subtitle_area_selected(self, rect):
        self.show(); self.activateWindow(); self.raise_(); self.screen_selector_overlay = None
        if rect and not rect.isNull() and rect.isValid():
            self.subtitle_bbox = rect; print(f"Área legenda definida: {rect}")
            self.status_label.setText(f"Status: Área legenda definida em {rect.x()},{rect.y()} [{rect.width()}x{rect.height()}]."); self.status_label.setStyleSheet("color: blue;")
            if self.subtitle_window is None: self.subtitle_window = SubtitleWindow(self.subtitle_bbox); self.subtitle_window.show()
            else: self.subtitle_window.update_geometry(self.subtitle_bbox); self.subtitle_window.show() if not self.subtitle_window.isVisible() else None
        else:
            print("Seleção área legenda cancelada/inválida.")
            if self.subtitle_bbox is None:
                 current_status = self.status_label.text()
                 if "Pronto" not in current_status and "definida" not in current_status: self.status_label.setText("Status: Seleção área legenda cancelada."); self.status_label.setStyleSheet("color: orange;")

    # --- Start/Stop Capture (Usa idiomas e chave/serviço da GUI) ---
    def start_capture(self):
        if self.capture_bbox is None or not isinstance(self.capture_bbox, dict) or self.capture_bbox.get('width',0) <= 0:
            QMessageBox.warning(self, "Área OCR Inválida", "Defina a área de captura de texto (OCR) usando 'Refinar Área OCR'.")
            return

        # Verifica se os idiomas selecionados são diferentes
        if self.selected_source_lang == self.selected_target_lang:
             QMessageBox.warning(self, "Idiomas Iguais", "O idioma de origem e destino são os mesmos. Nenhuma tradução será realizada.")
             # Poderia optar por não iniciar ou apenas avisar

        # --- Verificação API Selecionada ---
        service = self.selected_service
        if service == DEEPL:
            if not deepl: QMessageBox.critical(self, "Erro DeepL", "Biblioteca 'deepl' não instalada."); return
            if not self.prompt_for_deepl_key(force_prompt=False): QMessageBox.critical(self, "Erro DeepL", "Chave API DeepL necessária e não fornecida/inválida."); return
        elif service == GOOGLE:
            if not google_translate: QMessageBox.critical(self, "Erro Google", "Biblioteca 'google-cloud-translate' não instalada."); return
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                 ret = QMessageBox.warning(self, "Aviso Google", "Variável GOOGLE_APPLICATION_CREDENTIALS não definida.\nAutenticação pode falhar.\n\nDeseja continuar?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                 if ret == QMessageBox.No: return

        # --- Inicialização ---
        if self.subtitle_bbox and self.subtitle_window and not self.subtitle_window.isVisible(): self.subtitle_window.show()
        elif self.subtitle_bbox and self.subtitle_window is None: self.subtitle_window = SubtitleWindow(self.subtitle_bbox); self.subtitle_window.show()

        self.original_text.clear(); self.translated_text.clear();
        if self.subtitle_window: self.subtitle_window.update_text("...")
        self.status_label.setText(f'Status: Iniciando worker ({service}: {self.selected_source_lang} -> {self.selected_target_lang})...'); self.status_label.setStyleSheet("color: orange;")

        # Desabilita controles
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.select_window_button.setEnabled(False); self.refine_area_button.setEnabled(False); self.select_subtitle_area_button.setEnabled(False)
        self.service_selector.setEnabled(False)
        self.source_lang_selector.setEnabled(False) # Desabilita idiomas
        self.target_lang_selector.setEnabled(False) # Desabilita idiomas

        # Cria Worker com idiomas selecionados
        self.worker = Worker(
            capture_bbox=self.capture_bbox,
            ocr_lang_internal=self.selected_source_lang, # Passa código interno
            target_lang_internal=self.selected_target_lang, # Passa código interno
            interval_ms=CAPTURE_INTERVAL_MS,
            translation_service=service,
            libretranslate_url=LIBRETRANSLATE_URL,
            deepl_api_key=self.deepl_api_key,
            num_threads=NUM_TRANSLATION_THREADS
        )

        self.worker.last_accepted_ocr_text = ""; self.worker.last_displayed_request_id = -1; self.worker.translation_request_id = 0; self.worker.pending_originals.clear()
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True, name="WorkerThread")
        self.worker.update_text_signal.connect(self.update_text_boxes); self.worker.status_update_signal.connect(self.update_status); self.worker.error_signal.connect(self.show_error)
        self.worker_thread.start()

    def stop_capture(self):
        if self.worker: self.worker.stop()
        self.status_label.setText('Status: Parando captura...'); self.status_label.setStyleSheet("color: orange;")
        # Reabilita controles
        self.start_button.setEnabled(self.capture_bbox is not None and self.capture_bbox.get('width',0) > 0)
        self.stop_button.setEnabled(False)
        self.select_window_button.setEnabled(True)
        self.refine_area_button.setEnabled(self.selected_window_obj is not None)
        self.select_subtitle_area_button.setEnabled(True)
        self.service_selector.setEnabled(True)
        self.source_lang_selector.setEnabled(True) # Reabilita idiomas
        self.target_lang_selector.setEnabled(True) # Reabilita idiomas

    def _clear_worker_refs(self):
        self.worker = None; self.worker_thread = None; print("Referências GUI->worker limpas.")
        if 'parando' in self.status_label.text().lower(): self.update_status("Parado")

    # --- Métodos de Atualização de UI (sem alterações significativas) ---
    def update_text_boxes(self, original_ocr, translated_dialogue):
        max_history_items = 50; separator = "\n" + "═"*40 + "\n"; current_time_str = time.strftime('%H:%M:%S')
        current_original = self.original_text.toPlainText().split(separator)
        new_original = [f"--- OCR ({self.selected_source_lang} - {current_time_str}) ---\n{original_ocr}"] + current_original[:max_history_items-1]
        self.original_text.setPlainText(separator.join(new_original).strip()); self.original_text.verticalScrollBar().setValue(0)

        is_error = translated_dialogue.startswith(("[Erro", "[API Retornou Vazio]"))

        if not is_error and translated_dialogue:
            current_translated = self.translated_text.toPlainText().split(separator);
            service_name = self.worker.translation_service if self.worker else self.selected_service
            new_translated = [f"--- Tradução ({self.selected_target_lang} via {service_name} - {current_time_str}) ---\n{translated_dialogue}"] + current_translated[:max_history_items-1]
            self.translated_text.setPlainText(separator.join(new_translated).strip()); self.translated_text.verticalScrollBar().setValue(0)
            if self.subtitle_window and self.subtitle_window.isVisible(): self.subtitle_window.update_text(translated_dialogue)
        elif is_error:
            current_translated = self.translated_text.toPlainText().split(separator);
            error_msg_only = translated_dialogue.strip("[]");
            new_translated = [f"--- {error_msg_only} ({current_time_str}) ---"] + current_translated[:max_history_items-1]
            self.translated_text.setPlainText(separator.join(new_translated).strip()); self.translated_text.verticalScrollBar().setValue(0)
            if self.subtitle_window and self.subtitle_window.isVisible():
                error_label = "[Erro API]";
                if "DeepL" in translated_dialogue: error_label = "[Erro DeepL]"
                elif "Google" in translated_dialogue: error_label = "[Erro Google]"
                elif "Libre" in translated_dialogue: error_label = "[Erro Libre]"
                elif "Serviço" in translated_dialogue: error_label = "[Erro Serviço]"
                elif "Idioma" in translated_dialogue: error_label = "[Erro Idioma]"
                self.subtitle_window.update_text(error_label)

    def update_status(self, message):
        self.status_label.setText(f'Status: {message}')
        msg_lower = message.lower()
        if "parada" in msg_lower or "parado" in msg_lower:
             self.status_label.setStyleSheet("color: grey;")
             self.start_button.setEnabled(self.capture_bbox is not None and self.capture_bbox.get('width',0) > 0)
             self.stop_button.setEnabled(False); self.select_window_button.setEnabled(True)
             self.refine_area_button.setEnabled(self.selected_window_obj is not None)
             self.select_subtitle_area_button.setEnabled(True); self.service_selector.setEnabled(True)
             self.source_lang_selector.setEnabled(True); self.target_lang_selector.setEnabled(True) # Habilita idiomas
             if self.worker is not None or self.worker_thread is not None: QTimer.singleShot(50, self._clear_worker_refs)
        elif "erro" in msg_lower or "falha" in msg_lower or "inválida" in msg_lower or "não suportado" in msg_lower:
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            # Erros críticos que param a captura e desabilitam start
            critical_errors = ["inicializar", "capturar área", "chave api", "credenciais",
                               "conexão", "não instalada", "chave inválida", "idioma não é suportado"]
            if any(err in msg_lower for err in critical_errors):
                 if self.worker and self.worker._is_running: self.stop_capture()
                 QTimer.singleShot(50, self._clear_worker_refs)
                 self.start_button.setEnabled(False); self.stop_button.setEnabled(False)
                 self.service_selector.setEnabled(True) # Permite trocar serviço/chave
                 self.source_lang_selector.setEnabled(True) # Permite trocar idioma
                 self.target_lang_selector.setEnabled(True)
                 if "chave inválida" in msg_lower and self.selected_service == DEEPL:
                      self.deepl_api_key = None; self.save_current_config()
        elif "submetendo" in msg_lower or "inicializando" in msg_lower or "carregando" in msg_lower or "aviso:" in msg_lower or "filtrado" in msg_lower or "chamadas api" in msg_lower or "parando captura" in msg_lower: self.status_label.setStyleSheet("color: orange;")
        elif "exibida" in msg_lower or "exibido" in msg_lower: self.status_label.setStyleSheet("color: darkcyan;")
        elif "área ocr definida" in msg_lower or "pronto" in msg_lower:
             if "legenda definida" not in self.status_label.text() and "Erro" not in self.status_label.text(): self.status_label.setStyleSheet("color: green;")
        elif "janela selecionada" in msg_lower or "refine a área" in msg_lower: self.status_label.setStyleSheet("color: orange;")
        elif "legenda definida" in msg_lower: self.status_label.setStyleSheet("color: blue;")
        elif "ocr similar" in msg_lower: self.status_label.setStyleSheet("color: #808080;") # Cinza para similar
        else: self.status_label.setStyleSheet("color: green;") # Verde para capturando/status normal

    def show_error(self, message):
        self.update_status(f'ERRO: {message}'); print(f"ERRO GUI/Worker: {message}")
        critical_errors = ["Erro ao capturar área", "inválida", "Falha ao inicializar",
                           "Erro API: Conexão", "Chave API", "Credenciais", "não instalada",
                           "Chave Inválida", "Idioma não é suportado"] # Adicionado erro de idioma
        if any(err in message for err in critical_errors):
            if self.worker and self.worker._is_running: self.stop_capture()
            else:
                self._clear_worker_refs(); self.start_button.setEnabled(False); self.stop_button.setEnabled(False)
                self.select_window_button.setEnabled(True); self.refine_area_button.setEnabled(self.selected_window_obj is not None)
                self.select_subtitle_area_button.setEnabled(True); self.service_selector.setEnabled(True)
                self.source_lang_selector.setEnabled(True); self.target_lang_selector.setEnabled(True) # Habilita idiomas
            self.window_label.setText("Erro crítico! Verifique console/status."); self.window_label.setStyleSheet("color: red;")
            if "capturar área" in message or "inválida" in message: self.capture_bbox = None; self.start_button.setEnabled(False)
            if "Chave Inválida" in message and self.selected_service == DEEPL: self.deepl_api_key = None; self.save_current_config()

    # --- Close Event (Salva a configuração final) ---
    def closeEvent(self, event):
        print("Fechando aplicação...");
        self.save_current_config() # Salva estado final (serviço, chave, idiomas)
        if self.refinement_overlay: self.refinement_overlay.close()
        if self.screen_selector_overlay: self.screen_selector_overlay.close()
        if self.subtitle_window: self.subtitle_window.close()
        if self.worker:
            self.worker.stop()
            if self.worker_thread and self.worker_thread.is_alive():
                print("Aguardando thread worker..."); self.worker_thread.join(timeout=1.5)
                if self.worker_thread.is_alive(): print("Aviso: Thread worker não terminou.")
        print("Aplicação fechada.")
        event.accept()

# --- Execução Principal ---
if __name__ == '__main__':
    try:
        gpu_available = paddle.is_compiled_with_cuda()
        print(f"PaddlePaddle compilado com CUDA: {gpu_available}")
        if gpu_available:
            try: print(f"GPUs disponíveis: {paddle.device.cuda.device_count()}, Usando: {paddle.get_device()}")
            except Exception as e_gpu_info: print(f"Aviso: Detalhes GPU não obtidos ({e_gpu_info}).")
    except Exception as e_paddle_check: print(f"Erro verificar Paddle/CUDA: {e_paddle_check}")

    initial_config = load_config() # Carrega config ANTES da GUI

    # Avisos iniciais (opcional)
    if initial_config[CONFIG_KEY_SERVICE] == GOOGLE and google_translate and not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'): print("--- AVISO GOOGLE: GOOGLE_APPLICATION_CREDENTIALS não definida. ---")
    if initial_config[CONFIG_KEY_SERVICE] == DEEPL and deepl and not initial_config[CONFIG_KEY_DEEPL]: print("--- AVISO DEEPL: Nenhuma chave API salva. Será solicitada. ---")
    # Aviso sobre idioma OCR
    ocr_code_check = get_lang_code(initial_config[CONFIG_KEY_SOURCE_LANG], OCR_SERVICE)
    print(f"--- Idioma OCR inicial: {initial_config[CONFIG_KEY_SOURCE_LANG]} (Código para Paddle: {ocr_code_check}) ---")
    print(f"--- Idioma Tradução inicial: {initial_config[CONFIG_KEY_TARGET_LANG]} ---")


    app = QApplication(sys.argv)
    ex = OCRTranslatorApp(initial_config) # Passa config para a GUI
    sys.exit(app.exec_())
