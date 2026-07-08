# QSES Research Journal
Cumulative — each phase appended below the previous, never overwritten.

---

## Faz 1-2 (Unicode + Sample Validity)

**Hypothesis:** Projenin çalışmamasının tek nedeni Unicode encoding değildir; veri kalitesi sorunları da var.

**Experiment:** all_results.csv incelendi, open() çağrıları tarandı.

**Observation:** 1 open() çağrısı cp1254'de çöküyordu. CSV'nin %36'sı total_trades < 5 idi. profit_factor bazı satırlarda milyarlarca değer taşıyordu.

**Unexpected Findings:** profit_factor sonsuzluğu (sıfır kayıplı trade setleri) varlığı sürpriz oldu — win rate %100 olan satırlar yeşil heatmap hücresi olarak görünüyordu. Bu görsel olarak güçlü görünüyordu ama istatistiksel anlamı sıfırdı.

**Next Questions:**
- XU100'de sıfır trade neden? (FAZ 2'de cevaplanmıştı: cumulative VWAP)
- MIN_VALID_TRADES=10 doğru seçim mi, 15 daha iyi olmaz mıydı?
- `profit_factor = None` yerine `inf` göstermek kullanıcıya daha açık olur muydu?

---

## Faz 3 (TradingView Seed)

**Hypothesis:** TradingView'da elle optimize edilen parametreler, Python portu için iyi başlangıç noktası oluşturur ve optimizer convergence'ını hızlandırır.

**Experiment:** 3 parite seed'i `TV_REFERENCE_SEEDS` olarak eklendi. Optimizer trial-0 olarak seed'i denedi.

**Observation:** Seed sonuçları ve optimizer sonuçları `seed_vs_optimized_comparison.csv`'ye yazıldı. NQ1! seed'i WR=%80, optimizer benzer bölgede kaldı.

**Unexpected Findings:** XAUUSD seed'i negatif Sharpe üretti — Pine'da %68.1 WR iken Python'da çok daha az trade. Pine'ın session-reset VWAP'ı ve intrabar SL/TP execution'ı fundamentally farklı bir sonuç üretiyor. Parametre sorunu değil, execution model sorunu.

**Next Questions:**
- Pine intrabar SL/TP ile Python next-bar arasındaki fark kaça kadar P&L etkisi yapar?
- Rolling VWAP window boyutu XAUUSD için optimal ne olmalı?
- Optimizer seed'in yakınına convergence ediyor mu, yoksa tamamen farklı bir bölgeye mi gidiyor?

---

## Faz 4 (USOIL + EURUSD)

**Hypothesis:** EURUSD'nin volume=0 durumu OFI hesaplamalarını tamamen bozacak ve 0 sinyal üretecek.

**Experiment:** `has_volume` kontrolü eklendi, forex için range-based proxy devreye alındı.

**Observation:** AlgorithmA forex verisinde trade üretir hale geldi. `[VOLUME=0]` logu explicit olarak yazıldı.

**Unexpected Findings:** AlgorithmB'nin volume surge filtresi ATR expansion proxy ile ikame edildiğinde, sonuçlar volume-bazlı versiyona oldukça yakındı. Trend piyasalarında range ve volume birlikte hareket ettiği için proxy makul çalışıyor.

**Next Questions:**
- EURUSD için AlgorithmC (pure mean-reversion) AlgorithmA'dan daha uygun mu?
- Forex'te commission model farklı mı olmalı (spread-based)?
- 6 market arasında en düşük korelasyonlu çifti bulabilir miyiz?

---

## Faz 5 (Intrabar SL/TP + Quality Framework)

**Hypothesis:** Intrabar SL/TP execution, Pine Strategy Tester ile Python simülatörü arasındaki trade sayısı farkını kapatacak.

**Experiment:** `_check_exit_intrabar()` yazıldı. Gap-open logic eklendi. 14 unit test yazıldı.

**Observation:** Execution doğrulandı. Ancak trade sayısı farkı kapanmadı — XAUUSD'de hala 8 trade (Pine'da 47).

**Unexpected Findings:** Intrabar SL/TP eklenmesi trade SAYISINI artırmadı, çünkü sorun SL/TP execution'da değildi. Gerçek sorun şu: XAUUSD'de atr_tp=10xATR ile avg_hold=124 bar. Sadece 2500 barlık sentetik veriyle bu kadar uzun pozisyonlar birbirini bloke ediyor. Pine Strategy Tester ise 4000+ gerçek bar üzerinde çalışıyordu. Execution model değil, **veri uzunluğu** farkı.

**Next Questions:**
- Sentetik veri, gerçek market davranışını yeterince temsil etmiyor. Gerçek yfinance verisiyle bu fark kapanır mı?
- `atr_tp=10xATR` XAUUSD için Python'da optimal değil; optimizer bunu 3-5x'e çekecek mi?
- avg_hold'u kısaltmak için `exit_thresh` daha sıkıştırılabilir mi, ne kadar WR kaybı olur?

---

## Faz 6 (Rolling Walk-Forward + İstatistiksel Doğrulama)

**Hypothesis:** Walk-forward train/test ayrımı uygulandığında, daha önce
"stabil" görünen AlgorithmA konfigürasyonlarının çoğu overfitting belirtisi
gösterecek ve elenecek.

**Experiment:** 3 market (NQ1!, XU100, XAUUSD) × 4 model (sadece NQ1! için
tam, diğerleri M2 ile pilot) × 3 rolling walk × 5 optimizer trial/walk.
Monte Carlo (1000 permütasyon), Bootstrap (5000 örnek), Deflated Sharpe
Ratio hesaplandı.

**Observation:** 6 test edilen konfigürasyondan sadece 1'i (NQ1! M2) tüm
gate'leri geçti. Ancak o bile bootstrap p-value=0.149 ile Buy&Hold'dan
istatistiksel olarak ayırt edilemiyor.

**Unexpected Findings:**
1. `train_test_decay` formülü train Sharpe negatif olduğunda işaret
   değiştiriyor (XU100: decay=-100%, train=-2.21, test=0.00 — matematiksel
   olarak "iyileşme" gibi görünüyor ama aslında ikisi de kötü). Bu formülün
   bir sınır-durum zayıflığı — Faz 7'de düzeltilmeli.
2. n_trials=5 (zaman kısıtı nedeniyle düşürüldü) optimizer'ın gerçek
   potansiyelini göstermiyor olabilir. 50 trial ile NQ1! M0/M1/M3 belki
   10+ trade üretecek parametre bulabilirdi.
3. DSR (Deflated Sharpe Ratio) ham Sharpe'tan çok yüksek çıktı bazı
   konfigürasyonlarda (DSR=11.49 > SR_obs öncesi hesaplanan değer) —
   bu PSR formülünün küçük N (15 trial) ile garip davranması; gerçek
   N_trials büyüdükçe (288 backtest × 50 trial = 14400) DSR çok daha
   sert bir düzeltme uygulayacak.

**Next Questions:**
- n_trials=50 ile tam koşum yapılırsa NQ1! M0/M1/M3 sample gate'i geçer mi?
- train_test_decay formülü train_sharpe<0 durumunda nasıl düzeltilmeli?
  (önerilen: abs(train)+abs(test) tabanlı simetrik fark kullan)
- Gerçek piyasa verisiyle (sentetik değil) bu sonuçlar nasıl değişir?
- p=0.149 "anlamlı değil" sonucu, stratejinin gerçekten alfa üretmediğini
  mi gösteriyor, yoksa örneklem boyutu (13 trade) çok mu küçük?

---

## Faz 6 EK (RCA-7 Kapatma + Faz 7 Öncesi Teknik Borç Temizliği)

**Hypothesis:** Faz 7'de USOIL/EURUSD/SP500 walk-forward'u koşmadan önce,
`train_test_decay` formülündeki bilinen zayıflık (RCA-7) düzeltilmezse yeni
marketlerin PASS/ELIM kararları da aynı işaret hatasından etkilenecek.

**Experiment / Fix (RCA-7):**
- Issue: `train_test_decay = (avg_train - avg_test) / abs(avg_train)` —
  avg_train negatifken payda işareti korurken pay işareti bağımsız kalıyor,
  bu da küçük mutlak bozulmaların "%100 decay" gibi görünmesine yol açıyor
  (örn. avg_train=-0.2, avg_test=-0.4 → eski formülle decay=%100).
- Root Cause: Payda `abs(avg_train)` tek başına alınmış, `avg_test`'in payda
  içindeki katkısı yok. Bu simetrik olmayan payda, formülün büyüklüğünü
  keyfi biçimde küçültüp büyütebiliyor.
- Fix: `config/settings.py`'e `DECAY_EPSILON = 1e-6` eklendi.
  `optimization/walk_forward.py::_aggregate()` içinde formül
  `(avg_train - avg_test) / (abs(avg_train) + abs(avg_test) + DECAY_EPSILON)`
  olarak değiştirildi (Faz 6 günlüğünde önerilen fix). Ayrıca
  `compute_wf_score()` içindeki `cons_score` artık negatif decay
  (test > train) durumunda 100'ü aşmasın diye üst sınırla (min/max) sarmalandı.
- Validation: `tests/test_walk_forward_decay.py` — 6 yeni test (pozitif
  sağlık kontrolü, RCA-7'nin asıl regresyon senaryosu, sınırlılık [-1,1],
  train==test→decay=0, avg_train=avg_test=0 için sıfıra bölme yok,
  cons_score üst sınır). Tümü PASS.

**Unexpected Findings — N+1 ATR maddesi zaten kapalıydı:**
Dokümanın "Açık Problemler" listesinde "SL/TP seviyeleri N barının ATR'siyle
hesaplanıyor, N+1 ATR kullanılmalı" maddesi vardı. Kod incelemesinde
`algorithms/base.py` satır ~177'de girişin zaten `atr_entry = atrs[i + 1]`
(entry bar'ın kendi ATR'si) kullandığı görüldü — yani bu iş zaten yapılmış,
ancak doğrulayan bir test veya research_journal kaydı yoktu.
`tests/test_intrabar_execution.py::test_stop_loss_uses_entry_bar_atr_not_signal_bar_atr`
eklendi: sinyal barında (10) düşük volatilite, giriş barında (11) keskin
(ama aşağı yönde ihlal etmeyen) bir ATR sıçraması kurgulanıp, gerçekleşen
stop seviyesinin sinyal barının ATR'si değil giriş barının ATR'si ile
eşleştiği sayısal olarak kanıtlandı (exit tam olarak bar 13'te ve
exit_px == entry_bar ATR'sinden hesaplanan stop, ±0 tolerans). PASS.
Bu madde artık "açık teknik borç" listesinden çıkarılmalı; doküman/README
bu konuda güncel değildi.

**Diğer temizlik:**
- `requirements.txt` repoda mevcut değildi (README ve proje dokümanı ona
  atıfta bulunuyor) — oluşturuldu.
- `README.md` hâlâ v1.0 / "4 market × 192 kombinasyon" diyordu; gerçek kod
  6 market (`config/settings.py::MARKETS`) × 288 kombinasyonu destekliyor —
  v1.5'e güncellendi.

**Regression check:** Tüm test paketi 15 → 22 teste çıktı, hepsi PASS
(`pytest tests/ -v`, 0 skip, 0 fail).

**Next Questions:**
- Faz 7 walk-forward'u artık düzeltilmiş decay formülüyle USOIL/EURUSD/SP500
  üzerinde n_trials=50 ile tam koşumla çalıştırılmalı.
- RCA-6 (sentetik veri drift) ve gerçek yfinance verisiyle doğrulama hâlâ
  açık — Faz 7 gerçek veriyle mi yoksa yine sentetik pilotla mı başlamalı?

---

## Faz 7 (Başlangıç: Walk-Forward CLI Entegrasyonu)

**Hypothesis:** Walk-forward (Faz 6) hiçbir zaman `main.py`/`engine.py`'a
bağlanmamıştı — sadece `optimization/walk_forward.py` modülü olarak
mevcuttu ve doküman onu bile elle yazılan bir `python -c` snippet'iyle
çağırmayı öneriyordu. USOIL/EURUSD/SP500'ü gerçekten test edebilmek için
önce düzgün, tekrar edilebilir bir CLI giriş noktası gerekiyor.

**Experiment:**
- `config/settings.py::WALK_FORWARD_MARKETS` 3 marketten (NQ1!, XU100,
  XAUUSD) 6 markete genişletildi (SP500, USOIL, EURUSD eklendi).
- `main.py`'a `--walk-forward`, `--models`, `--n-walks` argümanları ve
  `run_walk_forward()` fonksiyonu eklendi: `DataFetcher` → her
  (market, algoritma, model) için `WalkForwardEngine.run()` →
  `reporting/walk_forward_reporter.py` ile CSV/MD çıktıları.
- Bug: `walk_forward_reporter.py`'deki `save_*` fonksiyonları
  `REPORTS_DIR`'i önceden oluşturmuyordu (ana `Reporter` sınıfı bunu
  `__init__`'te yapıyor ama walk-forward reporter'ın kendi `__init__`'i
  yok) — `results/reports/` yoksa `to_csv()` `OSError` fırlatıyordu.
  Fix: her `save_*` fonksiyonuna `_ensure_reports_dir()` guard'ı eklendi,
  ayrıca `main.py::run_walk_forward()` başında da `os.makedirs` çağrısı var.
- `optimizer.py` zaten `TV_REFERENCE_SEEDS`'te olmayan marketler için
  sorunsuz şekilde debug log basıp saf random search'e düşüyor — SP500/
  USOIL/EURUSD için ek kod gerekmedi (doğrulandı: `optimizer.py` satır 56-73).

**Validation:**
`DataFetcher.fetch` mock'lanarak (bu ortamda yfinance'a ağ erişimi yok)
sentetik bir OHLCV serisiyle `run_walk_forward()` uçtan uca çalıştırıldı:
SP500 × AlgorithmA × M0/M1, 3 walk, n_trials=3. Dört rapor dosyası da
(`walk_forward_detail.csv`, `walk_forward_summary.csv`,
`bootstrap_significance.csv`, `selection_rationale.md`) doğru üretildi,
acceptance-test tablosu (EK-4 formatı) konsola basıldı, hiçbir exception
fırlamadı. Tam `pytest tests/ -v` paketi (22 test) bu değişikliklerden
sonra da PASS.

**Remaining Difference / Not Yet Done:**
Bu ortamın ağı yfinance/yahoo finance'a kapalı (egress sadece pip/npm/github
kaynaklarına izinli) — bu yüzden GERÇEK piyasa verisiyle uçtan uca koşum
burada YAPILMADI, sadece mock veriyle kod yolu doğrulandı. USOIL/EURUSD/
SP500 için gerçek walk-forward sonuçları (pass_ratio, avg_sharpe, DSR, vb.)
hâlâ elde edilmedi — bu, ağ erişimi olan bir makinede
`python -m qses.main --walk-forward --markets SP500 USOIL EURUSD --n-trials 50`
komutuyla çalıştırılmalı.

---

## RCA-8: Cross-Market pass_ratio Birim Uyuşmazlığı (gerçek 288-koşum verisinde bulundu)

- **Issue:** Kullanıcının gerçek yfinance verisiyle çalıştırdığı ilk tam
  288-kombinasyon koşumunun `ranking_table.html` çıktısında "Markets"
  kolonu `8/6`, `19/6`, `20/6` gibi payda küçük paydan büyük değerler
  gösteriyordu — mantıksal olarak imkânsız bir oran (>1.0).
- **Symptoms:** `optimization/ranker.py::_score_group()` içinde
  `markets_tested = len(set(market for r in group))` (DAİMA ~6, distinct
  market sayısı) fakat `markets_passed = sum(1 for r in group if valid ve
  hard-filter geçti)` — bu bir SATIR/kombinasyon sayacı (market × timeframe
  × period, max 24), market sayacı değil.
- **Evidence:** AlgorithmA/M0 gerçek veride: XU100'de 4 kombinasyonun 4'ü de
  geçersiz (`zero_trades` / `insufficient_trades`) — yani XU100'de sıfır
  geçerli sonuç. Buna rağmen eski kod `markets_passed=8` (diğer 5 markette
  toplam 8 geçen SATIR), `markets_tested=6` → `pass_ratio=8/6=1.33` üretti;
  seçim eşiği (`>=0.5`) rahatça geçildi ve XU100'deki tam başarısızlık
  sonucu hiç etkilemedi.
- **Root Cause:** Payda "distinct market sayısı", pay "geçen satır sayısı"
  — iki farklı birim aynı orana bölünüyordu.
- **Impact:** Projenin temel tasarım ilkesi olan "Sadece 1 piyasada başarılı,
  diğerlerinde başarısız → elenir" (README, core felsefe) fiilen devre dışı
  kalıyordu: tek bir markette 3-4 kombinasyon başarılı olan bir konfigürasyon,
  diğer 5 markette tamamen başarısız olsa bile gate'i geçebiliyordu.
- **Fix:** `markets_passed` artık `len(set(market for r in group if valid ve
  hard-filter geçti))` — yani geçen SATIR değil, geçen DISTINCT market
  sayısı. Artık `pass_ratio` her zaman [0, 1] aralığında ve gerçekten
  "kaç market başarılı oldu" sorusuna cevap veriyor.
- **Validation:** `tests/test_ranker_cross_market_gate.py` — 4 yeni test:
  (1) tek marketin 4 kombinasyonu da geçse bile pass_ratio'nun 1/6'yı
  geçmediğini ve gate'in reddettiğini kanıtlıyor, (2) çoğunluk marketi
  geçince gate'in doğru şekilde kabul ettiğini, (3) aynı marketin birden
  fazla kombinasyonunun tek sayıldığını, (4) gerçek AlgorithmA/M0 XU100
  senaryosunu birebir sentetik olarak yeniden üretip markets_passed=5
  (8 değil) olduğunu doğruluyor.
- **Regression check:** Gerçek `all_results.csv` (226/288 geçerli,
  gerçek yfinance verisi) düzeltilmiş `Ranker` ile yeniden işlendi.
  Bu özel veri setinde TOP 5 SEÇİM DEĞİŞMEDİ (AlgorithmA M0/M1/M2/M3 +
  AlgorithmC M3) — ama raporlanan pass_ratio değerleri artık mantıklı
  (örn. AlgorithmA M0: 8/6=1.33 yerine 5/6=0.83; AlgorithmA M3: 19/6=3.17
  yerine 6/6=1.00). Farklı bir veri setinde bu hata seçimi gerçekten
  değiştirebilirdi (bkz. test senaryosu #1).
- **Remaining Difference:** Yok — fix tam kapsamlı, tüm test paketi
  (26 test) PASS.
