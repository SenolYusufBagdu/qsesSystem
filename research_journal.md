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
