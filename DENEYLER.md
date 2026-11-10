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

## E002 — Test sızıntısız seçim (orijinal protokol)

- **Tarih:** _(doldur)_
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
