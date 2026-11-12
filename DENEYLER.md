# Deney günlüğü — GTFN

Her deney için bir giriş. Deneyi **başlatmadan önce** üst kısmı (amaç, hipotez, ne değişti) doldur, **bittikten sonra** sonuç ve yorumu ekle.
Sayılar için `results/.../DENEY_KAYDI.md` dosyasına bak: çalıştırıcı onu otomatik doldurur. Bu dosya ise *neden* yaptığını ve *ne öğrendiğini* tutar.

Yeni giriş için en alttaki şablonu kopyala. Kimlikler sırayla gider: E001, E002, …

---

## E001 — Orijinal GTFN replikasyonu (orijinal protokol)

- **Tarih:** _(doldur)_
- **Amaç:** Yazarların kodunu değiştirmeden makaledeki sonuçları yeniden üretmek.
- **Kod:** orijinal repo `ahappyyang/GTFN`, commit `36ac947`. Yalnızca veri yolları değiştirildi (Drive'daki kopyada, Git'te izlenmiyor).
- **Protokol:** GTFN'nin kendi bölmesi (sınıf başına 30 eğitim pikseli, ≤60 pikselli sınıflarda 15), doğrulama kümesi yok. 200 epoch, seed 0–9. Parametreler README'deki gibi.
- **Nerede:** `GTFN_Colab.ipynb` · sonuçlar `hsi-gnn-PHD/GTFN/output/<veri>_seedNN.json` (yalnızca OA, AA, κ)
- **Sonuçlar (test, ort. ± std, n=10):**

  | Veri | OA | AA | κ | Makale OA |
  |---|---|---|---|---|
  | Indian Pines | 93,40 ± 0,96 | 96,12 ± 0,54 | 92,47 ± 1,09 | 94,00 |
  | Pavia University | 93,84 ± 1,63 | 94,34 ± 0,84 | 91,91 ± 2,10 | 95,14 |
  | Salinas | 96,42 ± 0,75 | 98,40 ± 0,34 | 96,02 ± 0,83 | 96,81 |

- **Yorum:** Makaledeki değerlere 0,4–1,3 puan yakın, replikasyon büyük ölçüde başarılı.
  **Ancak** en iyi epoch test kümesine göre seçiliyor (epoch 121, 131, …, 191, 200 arasından): sonuçlar iyimser olabilir.
- **Bilinen sorunlar:** test sızıntısı; karıştırılmış batch'lere ilk 64 örneğin komşu indeksleri veriliyor (değerlendirme de bu yüzden rastgele); süre, sınıf doğrulukları ve commit kaydedilmedi.
- **Sonraki adım:** → E002

---

## E002 — Test sızıntısız seçim (orijinal protokol)  reproducibility ve evaluation protocol analysis mpact of Test-Based Epoch Selection on GTFN Performance in Hyperspectral Image Classification

- **Tarih:** 11.11.2026
- **Soru:** Epoch seçimi test yerine doğrulama kümesiyle yapılırsa GTFN'nin sonucu ne kadar değişir?
- **Hipotez:** OA biraz düşer (tahminen 0,2–1 puan). Fark en çok, seed'ler arası oynaklığın yüksek olduğu Pavia'da görülür.
- **Ne değişti:** Yalnızca seçim yöntemi. Test havuzundan sınıf başına `min(10, kalan // 2)` doğrulama pikseli ayrıldı ve testten çıkarıldı.
  Eğitim kümesi, model, eğitim sırası ve değerlendirme takvimi E001 ile birebir aynı (CPU'da epoch epoch karşılaştırılarak doğrulandı).
- **Kod:** fork, commit `_(doldur: git log --oneline -1)_`
- **Nasıl:** `run_noleak_colab.ipynb`, 7. hücre
  ```
  python -u scripts/run_seeds.py --out_root .../results/GTFN_noleak \
      --datasets Indian PaviaU Salinas --seeds 0-9 --wandb_project hsi-gtfn \
      --exp_id E002 --note "Orijinal protokol, test sızıntısız seçim"
  ```
- **Nerede:** Drive `hsi-gnn-PHD/results/GTFN_noleak/` · W&B projesi `hsi-gtfn`
- **Sonuçlar:** _(DENEY_KAYDI.md'deki tabloyu buraya kopyala)_
- **Yorum:** _(hipotez doğrulandı mı? sızıntı farkı ne kadar? hangi veri setinde en büyük?)_
- **Sonraki adım:** _(ör. aynı düzenle eşit protokol bölmeleri → E003; indeks hatasının düzeltilmesi → E004)_

---

## Şablon

```
## E00X — Kısa başlık

- **Tarih:**
- **Soru / amaç:**
- **Hipotez:**
- **Ne değişti:** (bir önceki deneye göre tek fark ne?)
- **Kod:** fork, commit `...`
- **Nasıl:** (notebook + hücre ya da komut)
- **Nerede:** Drive klasörü · W&B projesi
- **Sonuçlar:**
- **Yorum:**
- **Sonraki adım:**
```
- **Tarih:** 2026-11-11
- **Kod:** fork, commit `e8275af`
- **Sonuçlar (test OA, ort. ± std, n=10):**

  | Veri | Val seçili | Son epoch | Sızıntılı | Fark | E001 |
  |---|---|---|---|---|---|
  | Indian | 93,12 ± 1,13 | 92,93 | 93,47 | +0,34 | 93,40 |
  | PaviaU | 93,03 ± 2,28 | 92,91 | 93,95 | +0,92 | 93,84 |
  | Salinas | 96,28 ± 0,77 | 96,12 | 96,39 | +0,11 | 96,42 |

- **Yorum:** Hipotez doğrulandı. Sızıntı OA'yı 0,1–0,9 puan şişiriyor ve en büyük etki
  PaviaU'da. Sızıntı kötü seed'leri gizliyor (PaviaU seed 8: 88,34 → 92,03) ve std'yi
  düşürüyor. Sızıntılı sütun E001 ile örtüşüyor, yani eğitim değişmemiş. Val seçimi ile son
  epoch arasında anlamlı fark yok (val kümesi küçük). Makaleye göre dürüst fark:
  −0,9 / −2,1 / −0,5 puan.
- **Sonraki adım:** E003 eşit protokol bölmeleri (splits/) ile GTFN · E004 indeks hatası düzeltmesi