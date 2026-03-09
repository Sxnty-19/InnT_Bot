import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes

# ==========================
# CONFIG
# ==========================

TELEGRAM_TOKEN = "8419596147:AAF1JO_gNT8ClNGdOSA6oxpKI6ZN7sGM41Y"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_MODEL = "SxntyM/innt_chat"

# ==========================
# CARGAR MODELO
# ==========================

print("Cargando modelo...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
)

model = PeftModel.from_pretrained(base_model, LORA_MODEL)

model.eval()

print("Modelo listo")

# ==========================
# GENERAR RESPUESTA
# ==========================

def generar_respuesta(pregunta):

    prompt = f"""### Instruction:
{pregunta}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    texto = tokenizer.decode(output[0], skip_special_tokens=True)

    # cortar solo la parte de la respuesta
    respuesta = texto.split("### Response:")[-1].strip()

    return respuesta


# ==========================
# COMANDO START
# ==========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    mensaje = """👋 Bienvenido al asistente inteligente del hotel.

Este bot responde preguntas sobre el hotel usando inteligencia artificial.

Para obtener mejores respuestas, escribe tus preguntas usando el formato:

¿Pregunta?

Ejemplos:
• ¿El hotel tiene piscina?
• ¿A qué hora es el check-in?
• ¿Dónde está ubicado el hotel?
• ¿Qué servicios ofrece el hotel?
• ¿El hotel incluye desayuno?

✍️ Consejo:
Escribe siempre preguntas completas y con signos de interrogación para obtener respuestas más precisas.

Cuando quieras, envía tu pregunta."""

    await update.message.reply_text(mensaje)


# ==========================
# RESPONDER MENSAJES
# ==========================

async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):

    pregunta = update.message.text

    print("Usuario:", pregunta)

    await update.message.chat.send_action("typing")

    respuesta = generar_respuesta(pregunta)

    print("Bot:", respuesta)

    await update.message.reply_text(respuesta)


# ==========================
# INICIAR BOT
# ==========================

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))

print("Bot activo")

app.run_polling()