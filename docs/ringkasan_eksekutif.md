# Ringkasan Eksekutif — SmokeFreeLab

> **Tujuan:** memperlihatkan satu kerangka analitik end-to-end untuk memutuskan
> eksperimen produk, alokasi media, dan segmentasi pelanggan di kategori
> smoke-free product (SFP), dengan semua angka dampak dibingkai dalam rupiah.

**Penulis:** Ahmad Zulfan (Az), Jakarta. Bilingual (Indonesian / English).
Brand paralel **Azul Analysis** untuk riset makro Indonesia.

---

## Satu kalimat

Dengan lima analisa inti — funnel, A/B test Bayesian, multi-touch attribution,
price elasticity, dan Marketing Mix Model — SmokeFreeLab memperlihatkan
potensi **uplift CLV tahunan sekitar Rp 27 miliar** dari perbaikan aktivasi
2 pp pada funnel SFP berskala 250 ribu registrasi per bulan.

---

## Lima angka yang penting

| # | Angka | Artinya |
|---|---|---|
| 1 | **Rp 27 miliar / tahun** | Incremental CLV bila aktivasi naik 2 pp. Rumus: 250K × 12 × 2 pp × Rp 450K LTV. Sumber: `notebooks/06_business_case.ipynb`. |
| 2 | **Rp 50 miliar / kuartal** | Asumsi total budget media TV + digital + trade (skenario rasional untuk kategori SFP di Jakarta). Dipakai sebagai titik awal rekomendasi re-alokasi MMM. |
| 3 | **Rp 1,4 miliar** | Estimasi incremental revenue per kuartal bila Rp 5 miliar dipindahkan dari TV ke trade promotion pada titik elbow saturasi saat ini. Sumber: `notebooks/09_mmm.ipynb`. |
| 4 | **48% CLV di top-decile** | Konsentrasi nilai pelanggan. Mengunci siapa yang tidak boleh churn. Sumber: `notebooks/08_clv_rfm.ipynb` (kurva Lorenz). |
| 5 | **~14% penurunan volume** | Efek kenaikan cukai dari Rp 3.000 ke Rp 3.800 per batang pada SKU elastis (|ε| ≈ 1,7). Sumber: `notebooks/07_price_elasticity.ipynb`. |

---

## Apa yang dibuktikan repositori ini

1. **Disiplin eksperimen.** Kerangka A/B frequentist + Bayesian dengan
   sample-size calculator, SRM gate, dan demonstrasi empiris α-inflation
   akibat peeking. Streamlit Experiment Designer (tiga tab: Planner,
   Readout, Peeking lab) siap dipakai Product Manager tanpa menulis kode.
2. **Atribusi multi-touch yang benar.** Implementasi Markov removal-effect
   (Anderl et al. 2016) dan Shapley coalitional (Dalessandro et al. 2012)
   sejajar dengan heuristik tradisional — kesenjangan alokasi antar-metode
   sendiri menjadi diagnostik.
3. **Metodologi MMM tanpa black box.** Bayesian Marketing Mix Model dengan
   custom adstock + Hill saturation di PyMC. Tidak memakai paket turnkey —
   setiap prior dapat dipertahankan dalam interview.
4. **Ekonomi FMCG — elasticity + CLV.** Log-log OLS + hierarchical Bayesian
   per kategori untuk price elasticity, BG/NBD + Gamma-Gamma untuk Customer
   Lifetime Value. Keduanya adalah kosakata default di tim brand dan
   finance FMCG.

---

## Cara membaca repositori ini dalam 5 menit

1. **Scroll README** sampai tabel *Featured deliverables*.
2. **Buka `notebooks/06_business_case.ipynb`** — semua angka rupiah di
   ringkasan ini turun dari sini.
3. **Buka `notebooks/09_mmm.ipynb`** — paling representatif untuk peran
   Data Scientist di FMCG.
4. **Jalankan `make run-app`** untuk melihat Experiment Designer langsung
   di browser.

---

## Kontak

Ahmad Zulfan (Az)
- LinkedIn: [linkedin.com/in/ahmadzulfan](https://www.linkedin.com/in/ahmadzulfan)
- GitHub: [azulcoder](https://github.com/azulcoder)
- Email: `infoman.xyz123@gmail.com`

---

> _Disclaimer: Simulasi media spend dan panel harga berbasis data sintetis
> yang kalibrasinya dirancang menyerupai pasar SFP di Indonesia. Yang
> di-ship di sini adalah **metodologi**, bukan klaim ground-truth ROI per
> channel atau elastisitas per SKU aktual._
