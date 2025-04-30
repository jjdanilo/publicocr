# **Tutorial Rápido: Executando o PublicOCR Real Time Translator**

Este guia mostra a forma mais rápida de executar o script, instalando tudo diretamente no seu Python principal.

https://youtu.be/sPlQGT31N4Q

## **Pré-requisitos:**

*   Instalar Python 3.12 com opção "add to path" marcada.

## **Passo 1: Preparar a Pasta e Instalar Dependências**

1. **Crie uma Pasta:** Crie uma pasta para o programa (ex: `publicocr`) em algum lugar fácil de achar.

2. **Coloque o Script:** Copie o arquivo `publicocr.py` para dentro dessa pasta.

3. **Abra o Terminal/Prompt de Comando/CMD NA PASTA:** Clique em alguma parte branca na pasta aberta com botão direito e clique em "Abrir no Terminal". Ou navegue até a pasta que você acabou de criar usando o comando `cd`. Exemplo:

   ```bash
   cd C:\Caminho\Para\Sua\Pasta\publicocr
   ```

4. **Instale as Bibliotecas:** Copie e cole o **comando inteiro** abaixo no seu terminal e pressione Enter. Isso instalará todas as bibliotecas necessárias diretamente no seu sistema. **Pode levar vários minutos.**

   ```bash
   pip install PyQt5==5.15.11 PyQt5-Qt5==5.15.2 PyQt5_sip==12.17.0 paddleocr==2.10.0 paddlepaddle==3.0.0 opencv-python==4.11.0.86 Pillow==11.2.1 numpy==2.2.5 scikit-image==0.25.2 pyclipper==1.3.0.post6 shapely==2.1.0 lmdb==1.6.2 requests==2.32.3 deepl==1.21.1 google-cloud-translate==3.20.2 google-api-core>=1.31.5,<3.0.0dev,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.* googleapis-common-protos>=1.56.2,<2.0.0dev proto-plus>=1.22.0,<2.0.0dev protobuf!=3.20.0,!=3.20.1,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.19.5 grpcio>=1.58.0,<2.0.0dev grpcio-status>=1.58.0,<2.0.0dev google-auth>=2.14.1,<3.0.0dev mss==10.0.0 PyGetWindow==0.0.9 PyRect==0.2.0 configparser PyYAML==6.0.2 tqdm==4.67.1 python-docx==1.1.2 lxml==5.4.0 fonttools>=4.24.0
   ```

   *(Observação: Este comando instala a versão CPU do `paddlepaddle`)*

## **Passo 3: Configurar Acesso às APIs (Se For Usar)**

*   **LibreTranslate(GRÁTIS):** Ver passo 5.
*   **DeepL(Mais fácil, com limite gratuito):** O programa pedirá sua chave de API na primeira vez que você usar o DeepL.
*   **Google Translate:** Você precisa configurar a variável de ambiente `GOOGLE_APPLICATION_CREDENTIALS` no seu sistema para apontar para o seu arquivo de chave `.json`.
    *   **Exemplo Rápido (Windows - Permanente):** Pesquise por "Variáveis de ambiente", vá em "Variáveis de Ambiente...", clique "Novo..." em "Variáveis de usuário", Nome: `GOOGLE_APPLICATION_CREDENTIALS`, Valor: `C:\caminho\completo\para\seu-arquivo-chave.json`. Dê OK em tudo e **reinicie o terminal/prompt**. (Pesquise online por guias mais detalhados se precisar).

## **Passo 4: Executar o Programa**

1. **Certifique-se** de que seu terminal/prompt ainda está aberto **na pasta** `publicocr`.

2. Digite o comando e pressione Enter:

   ```bash
   python publicocr.py
   ```

3. A interface do programa deve abrir.

4. **Primeira Execução (Modelos OCR):** Ao usar um idioma pela primeira vez no OCR, o programa pode precisar baixar modelos específicos. Aguarde o download.

### **Para Executar Novamente:**

Basta abrir o terminal/prompt, navegar até a pasta `publicocr` com `cd` e executar `python publicocr.py`.

### **Caso Queira Criar um Atalho**

Abra um editor de texto como VSCode ou Kate. Copie o código a seguir mudando o diretório do programa python e salve como .bat na área de trabalho.

```bash
@echo off
start pythonw.exe "C:\Users\Folder\publictranslator.py"
```

## **Passo 5: Instalar  Tradutor local gratuito LibreTranslate**

Ok, este é um tutorial rápido para configurar o LibreTranslate usando Docker Desktop, garantindo que ele tenha os modelos de idioma necessários para o seu script OCR Translator.

**Objetivo:** Rodar um servidor de tradução local (LibreTranslate) que seu script Python possa usar, sem depender de APIs externas pagas.

**Pré-requisitos:**

*   **Docker Desktop Instalado:** Se não tiver, baixe e instale do site oficial: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
*   **Windows 10/11 (Pro, Enterprise, Education ou Home com WSL 2)** ou **macOS** ou **Linux**.
*   **Conexão com a Internet** (para baixar a imagem do Docker).

---

**Tutorial Rápido: Configurando LibreTranslate com Docker**

**Passo 1: Verificar/Ativar Virtualização (Apenas Windows)**

O Docker Desktop precisa de virtualização. O WSL 2 é o método recomendado.

1. **Abra o PowerShell como Administrador:**

   *   Clique com o botão direito no menu Iniciar -> "Windows PowerShell (Admin)" ou "Terminal (Admin)".

2. **Execute os comandos abaixo** (um de cada vez, pressione Enter após cada um):

   ```powershell
   # Habilita o Subsistema Windows para Linux (WSL)
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   
   # Habilita a Plataforma de Máquina Virtual
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

3. **REINICIE O COMPUTADOR:** Isso é essencial para que as alterações tenham efeito.

4. **Após reiniciar, abra o PowerShell (normal, não precisa ser Admin) e defina o WSL 2 como padrão:**

   ```powershell
   wsl --set-default-version 2
   ```

   *(Se você receber um erro sobre a necessidade de atualizar o kernel do WSL, siga o link que ele fornecer para baixar e instalar a atualização).*

**Passo 2: Baixar e Executar o Contêiner LibreTranslate**

1. **Abra seu terminal normal** (PowerShell, cmd, Terminal do macOS/Linux).

2. **Copie e cole o comando abaixo** e pressione Enter. Este comando vai:

   *   Baixar a imagem do LibreTranslate (se ainda não tiver).
   *   Iniciar um contêiner chamado `libretranslate-ocr`.
   *   Disponibilizar o LibreTranslate na porta `5000` do seu computador.
   *   Instruir o LibreTranslate a carregar os modelos para os idiomas usados no seu script (isso pode levar um tempo na primeira vez e consumir mais RAM).
   *   Rodar em segundo plano (`-d`) e ser removido automaticamente ao parar (`--rm`).

   ```bash
   docker run --rm -d --name libretranslate-ocr -p 5000:5000 --env LT_LOAD_ONLY="en,pt-br,es,fr,de,it,ru,ja,zh,ko" libretranslate/libretranslate:latest
   ```

   *   **Aguarde:** Pode levar alguns minutos para baixar a imagem e iniciar/carregar os modelos.

**Passo 3: Verificar se Está Rodando**

1. **No terminal, digite:**

   ```bash
   docker ps
   ```

   Você deve ver uma linha mostrando o contêiner `libretranslate-ocr` com status "Up".

2. **Abra seu navegador de internet** e acesse:

   ```
   http://127.0.0.1:5000
   ```

   Você deve ver a interface web do LibreTranslate. Isso confirma que ele está funcionando!

**Passo 4: Usar com o OCR Translator**

1.  Execute seu script `ocr_translator.py` normalmente.
2.  Na interface do OCR Translator, certifique-se de que o serviço de tradução selecionado seja **"LibreTranslate"**.
3.  O script já está configurado para usar o endereço `http://127.0.0.1:5000`, então ele deve se conectar automaticamente ao contêiner Docker que você acabou de iniciar.
4.  Selecione os idiomas de origem e destino desejados (que você incluiu no comando `docker run`) e inicie a captura. As traduções agora usarão seu servidor local!

**OPICIONAL - Passo 5: Parar o Servidor LibreTranslate (Quando Terminar)**

1. Quando não precisar mais do servidor de tradução, volte ao seu terminal.

2. Digite o comando:

   ```bash
   docker stop libretranslate-ocr
   ```

   Isso vai parar e remover o contêiner (por causa da opção `--rm` usada no Passo 2).

**Para Iniciar Novamente:** Basta repetir o comando `docker run...` do Passo 2.

---

Pronto! Agora você tem um servidor LibreTranslate local rodando via Docker, pronto para ser usado pelo seu script Python, com suporte aos idiomas principais definidos.
