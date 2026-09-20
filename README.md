# بوت قصص الأطفال (Telegram)

يرسل قصة عربية مشوّقة (نص + صوت) كل 3 ساعات عبر **GitHub Actions**.

## الأوامر
- `/start` اشتراك
- `/stop` إلغاء
- `/now` قصة فورية (أثناء نافذة التشغيل القصيرة للمهمة)

## الأسرار المطلوبة
- `TELEGRAM_BOT_TOKEN`
- `GEMINI_API_KEY` (Aoede TTS + main prep / finalize fallback)
- `GEMINI_STORY_API_KEY` (اختياري — نهائي مع تشكيل كامل)
- `TALKING_MINI_API_KEY` (اختياري — Gemini ثانٍ للتحضير: استخراج / إعادة صياغة / مسودة عربية؛ soft-fail عند 403)
- `COHERE_API_KEY` (اختياري — احتياطي للتحضير)
- `SUBSCRIBER_CHAT_IDS` (اختياري، معرفات مفصولة بفواصل)

## مسار النماذج
1. **تحضير (prep):** Talking Mini Gemini → main Gemini → Cohere → Cloudflare Llama → BeautifulSoup  
   (توزيع الحمل: استخراج/ترجمة تفضّل Talking Mini؛ إعادة الصياغة تفضّل main)
2. **نهائي:** `GEMINI_API_KEY` → `TALKING_MINI_API_KEY` → `GEMINI_STORY_API_KEY` — قصة أطفال مع تشكيل كامل
3. **صوت:** Gemini TTS **Aoede** على `GEMINI_API_KEY`

## ملاحظة
المهمة تعمل كل 3 ساعات وتلتقط الأوامر لفترة قصيرة. للاشتراك: أرسل `/start` قرب وقت الإرسال، أو ضع معرف محادثتك في `SUBSCRIBER_CHAT_IDS`.

## الصوت والنص
- الراوية: Gemini TTS **Aoede** فقط (هادئة مستقرة، سرعة ثابتة).
- النص مع تشكيل كامل، يُرسل كملف `.txt` ثم الصوت.
