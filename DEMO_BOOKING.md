# عرض حي: حجز فحص صحي لعربة طعام

سيناريو العرض المباشر، **منفصل** عن مهمة التقييم الرسمية `book-inspection` في `data/eval/tasks.jsonl`. المهمة الرسمية لم تُعدَّل، حتى تبقى نتائج تقييمات سدايا قابلة للمقارنة.

البيانات اصطناعية من مدار: نورة، تصريح عربة الطعام `FTR-30311` (approved، حي الروضة).

## الطلب

```text
Hi, I own food truck permit FTR-30311. Please verify my permit, tell me what Madar's policy says about booking inspections, find a health inspection slot on Tuesday 3 or Wednesday 4 November before 12:00, and book it. I confirm the booking once the officer approves. ID ends 6654, mobile 0532236654.
```

## المسار المأمول (لم يُثبت بعد)

| الخطوة | الأداة | المصدر | موافقة؟ |
|---|---|---|---|
| التحقق من التصريح | `get_permit` | Local | لا |
| قراءة سياسة الحجز | `search_policies` | MCP | لا |
| البحث عن موعد | `list_slots` | Local | لا |
| الحجز | `book_inspection` | Local | **نعم: الموظف يكتب `y` أو `N`** |

الموعد الصحيح الوحيد هو `2026-11-04 09:00`: موعدا 3 نوفمبر الفارغان (12:00 و13:00) ليسا قبل 12:00.

**لا نفترض أن النموذج سيستدعي كل الأدوات بهذا الترتيب.** ما يحدث فعلًا يُثبت بالسطور `🔧 [local]`/`🔧 [mcp]` على الشاشة، وبسجل `artifacts/capstone/audit.jsonl`، وبنتيجة `check_booking.py`.

## الأمر (PowerShell، من أي مجلد)

```powershell
cd "C:\Users\Admin\Documents\SDAIA-Rafeeq-Capstone\agentic-ai\labs"; $env:Path = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe;$env:Path"; $env:UV_PYTHON_INSTALL_DIR = "$env:USERPROFILE\.uv\python"; $env:PYTHONIOENCODING = "utf-8"; [Console]::OutputEncoding = [Text.Encoding]::UTF8; uv run capstone/rafeeq.py ask "Hi, I own food truck permit FTR-30311. Please verify my permit, tell me what Madar's policy says about booking inspections, find a health inspection slot on Tuesday 3 or Wednesday 4 November before 12:00, and book it. I confirm the booking once the officer approves. ID ends 6654, mobile 0532236654."; uv run python capstone/check_booking.py FTR-30311
```

عند ظهور `🛡️  Allow book_inspection(permit_id=FTR-30311, slot=...)? [y/N]` اكتب `y` للموافقة أو `N` للرفض.

## ما نتحقق منه بعد التشغيل

- **عند الموافقة:** `check_booking.py` يُظهر الحجز في قاعدة الوكيل `artifacts/capstone/db/` فقط. لا يظهر في نسخة MCP ولا في بيانات التدريب.
- **عند الرفض:** لا يوجد أي حجز.
- **في الحالتين:** لا تظهر الهوية ولا الجوال على الشاشة (عدا السؤال الذي كتبناه في الأمر نفسه) ولا في `audit.jsonl`.

شغّل `check_booking.py` **مباشرة** بعد التجربة، لأن التشغيل التالي يبدأ من نسخة جديدة من البيانات. التجربة تستهلك نحو 4 إلى 6 طلبات من الحصة المجانية.
