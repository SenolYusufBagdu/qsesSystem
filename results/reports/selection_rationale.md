# QSES Walk-Forward Selection Rationale (Faz 6)

Markets: NQ1!, XU100, XAUUSD | Algorithm: AlgorithmA | Walks: 3 | Trials/walk: 5

---

## NQ1! — AlgorithmA M2 (SELECTED)

M2 secildi cunku diger 3 NQ1! modeli (M0,M1,M3) walk-forward'da hep
ayni paterne dustu: walk 1-2'de 0-1 trade (yetersiz), walk 3'te 5 trade
ile guclu sonuc (test Sharpe=3.84). Bu pattern, M0/M1/M3'un cok siki
parametrelerle (yuksek threshold, dar OFI penceresi) az sinyal uretmesinden
kaynaklaniyor; total_test_trades 6-7 ile MIN_VALID_TRADES=10 esiginin altinda
kaliyor, dolayisiyla otomatik ELENiYOR (Sample Gate FAIL), Sharpe degerleri
ne kadar yuksek gorunse de (DSR=11.49-11.67) istatistiksel olarak guvenilmez.

M2 (agresif model, dusuk threshold) walk 1-2'de de 4'er trade uretti,
toplam 13 trade ile Sample Gate'i (>=10) geciyor. Ancak decay=+39.9% > 0
demek train Sharpe (2.13 ort.) test Sharpe'tan (1.28) yuksek -- klasik
overfitting sinyali, ama MAX_SHARPE_DECAY=50% esiginin altinda kaldigi
icin Decay Gate'i de geciyor (sinirda).

KRITIK UYARI: p-value=0.149 > 0.05 -- M2'nin Sharpe'i Buy&Hold'dan
istatistiksel olarak ayirt edilemiyor. "PASS" etiketi sadece gate
esiklerini gecmesinden geliyor, gercek alfa varligini KANITLAMIYOR.

---

## NQ1! — M0, M1, M3 (ELIMINATED)

Ucu de ayni kok nedenden eleniyor: trades=6-7 < MIN_VALID_TRADES=10.
Walk 1-2'de optimizer 5 trial ile (kucuk arama uzayi) sinyal uretemeyen
parametre setleri buluyor -- bu n_trials=5'in cok dusuk olmasinin bir
sonucu olabilir, gercek 50 trial ile farkli sonuc cikabilir (bkz. Limitations).

---

## XU100 — M2 (ELIMINATED)

Walk 1-2-3'un ucu de test_sharpe=0.00 -- her walkte test setinde trade
kapanmiyor ya da kapanan trade'lerin pnl toplami sifira yakin. Train
Sharpe ortalamasi NEGATIF (-2.21), yani optimizer train'de bile iyi
parametre bulamadi. decay=-100% (formul: (avg_train-avg_test)/abs(avg_train),
train negatif oldugunda decay isareti degisiyor -- bu formulun bir
sinir-durum zayifligi, asagida RCA'da ele alindi). Worst walk DD=-26.1%,
bu tek basina kabul edilemez bir risk seviyesi.

ROOT CAUSE: n_trials=5 cok dusuk bir arama. XU100'un trending yapisi
(Faz 1-2 RCA'da tespit edilen VWAP/rolling sorunlari) ile birlesince
optimizer'in 5 trial'da iyi bir bolge bulmasi neredeyse imkansiz.

---

## XAUUSD — M2 (ELIMINATED)

Decay=+100% (train Sharpe ortalamasi pozitif 0.23, test Sharpe=0.00 --
train'de cok zayif da olsa pozitif sinyal var ama test'e hic tasinmiyor).
Bu, Faz 3-4 RCA'larinda tespit edilen "atr_tp=10xATR ile uzun tutulum"
sorununun walk-forward baglaminda tekrar ortaya cikmasi: kisa test
pencerelerinde (833 bar/walk) bu kadar genis TP'li pozisyonlar nadiren
kapanabiliyor.

---

## Genel Sonuc

6 konfigurasyondan sadece 1'i (NQ1! M2) tum gate'leri gecti, ve o bile
istatistiksel anlamllik testinde basarisiz (p=0.149). Bu, sistemin
su anki haliyle **walk-forward'da guvenilir alfa uretmedigini** gosteriyor.
Bu beklenmedik degil -- dogru calisan bir dogrulama surecinin sonucu budur.
